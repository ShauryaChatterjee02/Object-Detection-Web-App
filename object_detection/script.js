document.getElementById('uploadBtn').addEventListener('change', function() {
    var file = this.files[0];
    if (file) {
        var reader = new FileReader();
        reader.onload = function(e) {
            var output = document.getElementById('output');
            output.innerHTML = '<img src="' + e.target.result + '" alt="Uploaded Image" style="max-width: 100%;">';
        }
        reader.readAsDataURL(file);
    }
});
