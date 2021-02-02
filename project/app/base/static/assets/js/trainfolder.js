function setFolder(data){
	
	
$('#train-input').change(handleFolderSelect(data));	
	
}


function handleFolderSelect (e,data) {
    var files = e.target.files;
    if (files.length < 1) {
        // alert('select a file...');
        return;
    }
    var file = files[0];
    var reader = new FileReader();
    reader.onload = onFileLoaded;
    reader.readAsDataURL(file);
	reader.addEventListener("load", e => {
		var dataFromChunkG = JSON.parse(reader.result);
		// alert(dataFromChunkG.nodes);
	});	
	reader.readAsText(file);
	
	var fullPath = document.getElementById('train-input').value;
	document.getElementById("trainpath").innerHTML = fullPath
	if (fullPath) {
		var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
		var filename = fullPath.substring(startIndex);
		if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) {
			filename = filename.substring(1);
		}
	}
	
}


$(function () {
	
    $('#train-input').click();

    $('#train-input').change(handleFileSelect);
});