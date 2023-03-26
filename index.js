

window.onload = function () {
    var fileInput = document.getElementById('fileInput');
    var fileDisplayArea = document.getElementById('fileDisplayArea');

    fileInput.addEventListener('change', function (e) {
        var file = fileInput.files[0];
        var textType = /text.*/;

        if (file.type.match(textType)) {
            var reader = new FileReader();

            reader.onload = function (e) {
                fileDisplayArea.innerText = reader.result;
            }

            reader.readAsText(file);
            // const data = { name: 'Ronn', age: 27 };              //sample json
            // const a = document.createElement('a');
            // const blob = new Blob([JSON.stringify(reader.result)]);
            // a.href = URL.createObjectURL(blob);
            // a.download = 'sample-data.csv';                     //filename to download
            // a.click();

        }
        else {
            fileDisplayArea.innerText = "File not supported!"
        }
    });
}
