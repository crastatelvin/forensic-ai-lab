 // Basic interactivity for the futuristic UI
document.addEventListener('DOMContentLoaded', () => {
    console.log('Forensic Toolkit Loaded');

    // Add hover effect for buttons
    const buttons = document.querySelectorAll('.nav-btn, .module-btn');
    buttons.forEach(btn => {
        btn.addEventListener('mouseover', () => {
            btn.style.boxShadow = '0 0 15px #00d4ff';
        });
        btn.addEventListener('mouseout', () => {
            btn.style.boxShadow = 'none';
        });
    });

    // Form submission feedback
    const form = document.querySelector('.upload-form');
    if (form) {
        form.addEventListener('submit', () => {
            console.log('Analyzing fingerprints...');
        });
    }
});

document.addEventListener('DOMContentLoaded', () => {
    console.log('Forensic Toolkit Loaded');

    // Enhance module icons hover effect
    const moduleIcons = document.querySelectorAll('.module-icon');
    moduleIcons.forEach(icon => {
        icon.addEventListener('mouseover', () => {
            icon.style.filter = 'brightness(1.2)';
        });
        icon.addEventListener('mouseout', () => {
            icon.style.filter = 'brightness(1)';
        });
    });

    // Chatbot functionality
    const chatForm = document.getElementById('chatForm');
    const chatContainer = document.getElementById('chatContainer');
    const userInput = document.getElementById('userInput');

    if (chatForm) {
        chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const message = userInput.value.trim();
            if (message) {
                // Add user message
                const userMessage = document.createElement('div');
                userMessage.classList.add('chat-message', 'user-message');
                userMessage.textContent = message;
                chatContainer.appendChild(userMessage);

                // Simulate Grok response (mock API)
                setTimeout(() => {
                    const grokMessage = document.createElement('div');
                    grokMessage.classList.add('chat-message', 'grok-message');
                    grokMessage.textContent = `Grok: Analyzing... "${message}". How can I assist further?`; // Placeholder response
                    chatContainer.appendChild(grokMessage);
                    chatContainer.scrollTop = chatContainer.scrollHeight; // Scroll to bottom
                }, 1000); // Simulate delay

                userInput.value = '';
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        });
    }
});