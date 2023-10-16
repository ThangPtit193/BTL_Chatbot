const msg_input = document.getElementById('msg_input')
const server_url = `${window.location.protocol}//${window.location.host}`
const send_button = document.getElementById("send_button")
const csrf_token = document.getElementsByName('csrfmiddlewaretoken')[0]

function getCurrentTimestamp() {
    return new Date();
}

function renderMessageToScreen(args) {
    // local variables
    let displayDate = (args.time || getCurrentTimestamp()).toLocaleString('en-IN', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
    });
    let messagesContainer = $('.messages');

    // init element
    let message = $(`
	<li class="message ${args.message_side}">
		<div class="avatar"></div>
		<div class="text_wrapper">
			<div class="text">${args.text}</div>
			<div class="timestamp">${displayDate}</div>
		</div>
	</li>
	`);

    // add to parent
    messagesContainer.append(message);

    // animations
    setTimeout(function () {
        message.addClass('appeared');
    }, 0);
    messagesContainer.animate({scrollTop: messagesContainer.prop('scrollHeight')}, 300);
}

function showUserMessage(message, datetime) {
    renderMessageToScreen({
        text: message,
        time: datetime,
        message_side: 'right',
    });
}

function showBotMessage(message, datetime) {
    renderMessageToScreen({
        text: message,
        time: datetime,
        message_side: 'left',
    });
}

send_button.onclick = async function () {
    // Lấy và hiển thị tin nhắn từ người dùng
    const userMessage = $('#msg_input').val();
    showUserMessage(userMessage);

    // Gọi hàm chatbot_response và hiển thị kết quả
    try {
        const botMessage = await chatbot_response();
        showBotMessage(botMessage);
    } catch (error) {
        console.error(error);
    }
    $('#msg_input').val('');
};

function chatbot_response(length = 20) {
    return new Promise((resolve, reject) => {
        const description = $('#msg_input').val();

        $.ajax({
            type: "POST",
            url: server_url,
            headers: {
                "X-CSRFToken": csrf_token.value
            },
            data: {
                "input": description,
            },
            success: function (result) {
                const output = result['answer'];
                resolve(output);
            },
            error: function (error) {
                reject(error);
            },
            dataType: "json"
        });
    });
}
