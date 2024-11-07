pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'credit-fraud-detector'
        DOCKER_TAG = "${BUILD_NUMBER}"
        // Use environment variables for Docker Hub
        DOCKER_HUB_USERNAME = credentials('DOCKER_HUB_USERNAME')
        DOCKER_HUB_PASSWORD = credentials('DOCKER_HUB_PASSWORD')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Run Tests') {
            steps {
                sh '''
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                    python -m pytest tests/
                '''
            }
        }
        
        stage('Train Model') {
            steps {
                dir('model_training') {
                    script {
                        sh '''
                            docker build -t model-trainer .
                            docker run -v $(pwd)/models:/app/models model-trainer
                        '''
                    }
                }
            }
            post {
                success {
                    archiveArtifacts artifacts: 'model_training/models/*.pkl'
                }
            }
        }
        
        stage('Build Web App') {
            steps {
                dir('credit_fraud_app') {
                    sh 'mkdir -p app/models'
                    sh 'cp ../model_training/models/model.pkl app/models/'
                    
                    script {
                        sh "docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} ."
                    }
                }
            }
        }
        
        stage('Run Security Scan') {
            steps {
                sh "trivy image ${DOCKER_IMAGE}:${DOCKER_TAG}"
            }
        }
        
        stage('Push to Registry') {
            steps {
                script {
                    // Login to Docker Hub using environment variables
                    sh '''
                        echo $DOCKER_HUB_PASSWORD | docker login -u $DOCKER_HUB_USERNAME --password-stdin
                        
                        # Tag and push the image
                        docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_HUB_USERNAME}/${DOCKER_IMAGE}:${DOCKER_TAG}
                        docker push ${DOCKER_HUB_USERNAME}/${DOCKER_IMAGE}:${DOCKER_TAG}
                        
                        docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_HUB_USERNAME}/${DOCKER_IMAGE}:latest
                        docker push ${DOCKER_HUB_USERNAME}/${DOCKER_IMAGE}:latest
                    '''
                }
            }
        }
        
        stage('Deploy') {
            steps {
                script {
                    sh """
                        sed -i 's|image: credit-fraud-app:.*|image: ${DOCKER_HUB_USERNAME}/${DOCKER_IMAGE}:${DOCKER_TAG}|' docker-compose.yml
                        docker-compose up -d
                    """
                }
            }
        }
    }
    
    post {
        always {
            sh """
                docker rmi ${DOCKER_IMAGE}:${DOCKER_TAG} || true
                docker rmi ${DOCKER_HUB_USERNAME}/${DOCKER_IMAGE}:${DOCKER_TAG} || true
            """
        }
        success {
            slackSend(
                color: 'good',
                message: "Build #${BUILD_NUMBER} - Success! New version deployed."
            )
        }
        failure {
            slackSend(
                color: 'danger',
                message: "Build #${BUILD_NUMBER} - Failed! Check logs for details."
            )
        }
    }
}