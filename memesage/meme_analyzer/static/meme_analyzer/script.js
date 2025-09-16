// Function to get CSRF token from cookies for Django
function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    const cookies = document.cookie.split(';');
    for (let cookie of cookies) {
      cookie = cookie.trim();
      if (cookie.startsWith(name + '=')) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

const csrftoken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');


document.addEventListener('DOMContentLoaded', function () {
  const fileInput = document.getElementById('file-upload');
  const csrftoken = getCookie('csrftoken');

  if (!csrftoken) {
    console.warn('CSRF token not found in cookies.');
  }

  fileInput.addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);

    fetch('/api/meme/', {
      method: 'POST',
      headers: { 'X-CSRFToken': csrftoken },
      body: formData,
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log('Received data:', data);

        const tagsContainer = document.getElementById('tags-list');
        tagsContainer.innerHTML = '';

        let label = '';
        if (data.predicted_label && typeof data.predicted_label === 'string') {
          label = data.predicted_label;  // Use the label as is
        }

        if (label) {
          const span = document.createElement('span');
          span.className = 'tag-label';
          span.textContent = label;
          tagsContainer.appendChild(span);
        }

        document.getElementById('ocr-result').textContent = data.extracted_text || '';
      })
      .catch(error => {
        console.error('Error:', error);
        document.getElementById('ocr-result').textContent = 'Error analyzing meme. Please try again.';
      });
  });
});

