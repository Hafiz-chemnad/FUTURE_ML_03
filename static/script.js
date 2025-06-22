document.getElementById('chat-form').addEventListener('submit', async function(event) {
    event.preventDefault(); // Prevent default form submission

    const userInput = document.getElementById('user-input');
    const userMessage = userInput.value.trim();

    if (userMessage === '') {
        return; // Don't send empty messages
    }

    // Add user message to chat box
    addMessage(userMessage, 'user-message');
    userInput.value = ''; // Clear input field

    try {
        const formData = new URLSearchParams();
        formData.append('user_message', userMessage);

        const response = await fetch('/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: formData.toString()
        });

        if (!response.ok) {
            // If the server response is not OK (e.g., 500 error)
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        const botResponse = data.response;

        // Add bot message to chat box
        addMessage(botResponse, 'bot-message');

    } catch (error) {
        console.error('Error fetching chat response:', error);
        // Display a user-friendly error message in the chat
        addMessage("Oops! I'm having trouble processing that. Please try again later.", 'bot-message');
    }
});

function addMessage(text, className) {
    const chatBox = document.getElementById('chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);
    messageDiv.textContent = text;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the bottom
}

// Optional: Automatically focus on the input field when the page loads
window.onload = function() {
    document.getElementById('user-input').focus();
};
