const chatForm = document.getElementById('chatForm');
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Get user input
    const userMessage = userInput.value.trim();
    if (!userMessage) return;

    // Display user message
    addMessage(userMessage, 'user-message');

    // Clear input
    userInput.value = '';

    // Add "thinking" indicator
    const thinkingMessage = addMessage('Thinking...', 'thinking-message');

    // Send message to server and handle response
    try {
        const response = await fetch('/ai_analysis/ai_analyzer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_input: userMessage }),
        });

        if (!response.ok) {
            throw new Error('Failed to fetch response from server.');
        }

        const data = await response.json();
        thinkingMessage.remove(); // Remove "thinking" indicator
        addMessage(data.response, 'grok-message'); // Display server response
    } catch (error) {
        thinkingMessage.remove();
        addMessage(`Error: ${error.message}`, 'grok-message');
    }
});

function addMessage(message, className) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${className}`;
    messageDiv.innerHTML = `<span class="message-text">${message}</span>`;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return messageDiv;
}

document.addEventListener('DOMContentLoaded', () => {
    const chatbotToggle = document.getElementById('chatbot-toggle');
    const chatbotContainer = document.getElementById('chatbot-container');
    const closeChatbot = document.getElementById('close-chatbot');

    chatbotToggle.addEventListener('click', () => {
        chatbotContainer.classList.toggle('hidden');
    });

    closeChatbot.addEventListener('click', () => {
        chatbotContainer.classList.add('hidden');
    });
});
