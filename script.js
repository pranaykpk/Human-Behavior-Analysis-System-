document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const detectionToggle = document.getElementById('detectionToggle');
    const toggleLabel = document.getElementById('toggleLabel');
    const videoFeed = document.getElementById('videoFeed');
    const emotionIcon = document.getElementById('emotionIcon');
    const emotionText = document.getElementById('emotionText');
    const confidenceText = document.getElementById('confidenceText');
    const confidenceFill = document.getElementById('confidenceFill');
    const expressionChip = document.getElementById('expressionChip');
    const drowsyAlert = document.getElementById('drowsyAlert');
    const faceMarkingsToggle = document.getElementById('faceMarkingsToggle');
    
    // State
    let detectionActive = false;
    let faceMarkingsActive = false;
    let videoImg = null;
    let resultsFetchInterval = null;
    
    // Toggle detection
    detectionToggle.addEventListener('change', async function() {
        try {
            // Show loading animation
            videoFeed.innerHTML = '<div class="loading-animation"></div>';
            
            const response = await fetch('http://localhost:5000/api/toggle_detection', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    active: this.checked
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'started' || data.status === 'stopped') {
                detectionActive = this.checked;
                updateUI();
            }
        } catch (error) {
            console.error('Error toggling detection:', error);
            alert('Error connecting to the server. Please make sure the backend is running.');
            this.checked = false;
            detectionActive = false;
            updateUI();
        }
    });
    
    // Toggle face markings
    faceMarkingsToggle.addEventListener('click', async function() {
        try {
            faceMarkingsActive = !faceMarkingsActive;
            
            const response = await fetch('http://localhost:5000/api/toggle_face_markings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    show: faceMarkingsActive
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                updateFaceMarkingsButton();
            }
        } catch (error) {
            console.error('Error toggling face markings:', error);
            faceMarkingsActive = false;
            updateFaceMarkingsButton();
        }
    });
    
    // Update face markings button state
    function updateFaceMarkingsButton() {
        if (faceMarkingsActive) {
            faceMarkingsToggle.classList.add('active');
            faceMarkingsToggle.innerHTML = `
                <span class="material-icon">face</span>
                <span>Hide Face Markings</span>
            `;
        } else {
            faceMarkingsToggle.classList.remove('active');
            faceMarkingsToggle.innerHTML = `
                <span class="material-icon">face</span>
                <span>Show Face Markings</span>
            `;
        }
    }
    
    // Update UI based on detection state
    function updateUI() {
        toggleLabel.textContent = detectionActive ? 'Detection Active' : 'Detection Inactive';
        
        // Enable/disable face markings toggle based on detection state
        faceMarkingsToggle.disabled = !detectionActive;
        
        if (detectionActive) {
            // Load video feed
            videoFeed.innerHTML = '';
            videoImg = document.createElement('img');
            videoImg.src = `http://localhost:5000/video_feed?t=${Date.now()}`;
            videoImg.alt = 'Live Detection Feed';
            videoImg.style.maxWidth = '100%';
            videoImg.style.maxHeight = '100%';
            videoImg.style.objectFit = 'contain';
            
            videoImg.onerror = function(e) {
                console.error('Error loading video feed:', e);
                setTimeout(() => {
                    if (detectionActive) {
                        videoImg.src = `http://localhost:5000/video_feed?t=${Date.now()}`;
                    }
                }, 1000);
            };
            
            videoFeed.appendChild(videoImg);
            
            // Start fetching results
            fetchResults();
            resultsFetchInterval = setInterval(fetchResults, 1000);
        } else {
            // Clear video feed
            videoFeed.innerHTML = '<p>Toggle the switch to start detection</p>';
            videoImg = null;
            
            // Reset face markings state
            faceMarkingsActive = false;
            updateFaceMarkingsButton();
            
            // Reset result displays
            emotionIcon.className = 'icon icon-neutral';
            emotionText.textContent = 'None';
            confidenceText.textContent = 'Confidence: 0.0%';
            confidenceFill.style.width = '0%';
            expressionChip.textContent = 'None';
            expressionChip.className = 'chip chip-neutral';
            drowsyAlert.className = 'alert alert-success';
            drowsyAlert.innerHTML = `
                <span class="icon icon-check"></span>
                <span><strong>Alert and Attentive</strong></span>
            `;
            
            // Stop fetching results
            if (resultsFetchInterval) {
                clearInterval(resultsFetchInterval);
                resultsFetchInterval = null;
            }
        }
    }
    
    // Fetch detection results
    async function fetchResults() {
        try {
            const response = await fetch('http://localhost:5000/api/detection_status');
            const data = await response.json();
            
            // Update emotion with animation
            const oldEmotion = emotionText.textContent;
            if (oldEmotion !== data.emotion) {
                emotionText.style.opacity = 0;
                setTimeout(() => {
                    emotionText.textContent = data.emotion;
                    emotionText.style.opacity = 1;
                }, 300);
            } else {
                emotionText.textContent = data.emotion;
            }
            
            confidenceText.textContent = `Confidence: ${(data.emotion_prob * 100).toFixed(1)}%`;
            confidenceFill.style.width = `${data.emotion_prob * 100}%`;
            
            // Update emotion icon
            emotionIcon.className = 'icon';
            if (['Happy', 'Surprise'].includes(data.emotion)) {
                emotionIcon.classList.add('icon-happy');
            } else if (['Angry', 'Disgust', 'Fear', 'Sad'].includes(data.emotion)) {
                emotionIcon.classList.add('icon-sad');
            } else {
                emotionIcon.classList.add('icon-neutral');
            }
            
            // Update expression
            expressionChip.textContent = data.expression;
            expressionChip.className = 'chip';
            if (data.expression === 'Smile') {
                expressionChip.classList.add('chip-success');
            } else if (data.expression === 'Yawn') {
                expressionChip.classList.add('chip-warning');
            } else {
                expressionChip.classList.add('chip-neutral');
            }
            
            // Update drowsiness
            if (data.drowsy) {
                drowsyAlert.className = 'alert alert-error pulse';
                drowsyAlert.innerHTML = `
                    <span class="icon icon-warning"></span>
                    <span><strong>DROWSY ALERT!</strong></span>
                `;
            } else {
                drowsyAlert.className = 'alert alert-success';
                drowsyAlert.innerHTML = `
                    <span class="icon icon-check"></span>
                    <span><strong>Alert and Attentive</strong></span>
                `;
            }
            
            // Update face markings state if it changed from backend
            if (faceMarkingsActive !== data.show_face_markings) {
                faceMarkingsActive = data.show_face_markings;
                updateFaceMarkingsButton();
            }
            
        } catch (error) {
            console.error('Error fetching detection status:', error);
        }
    }
});