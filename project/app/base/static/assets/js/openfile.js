/* const {dialog} = require('electron').remote;

document.querySelector('#selectBtn').addEventListener('click', function (event) {
    dialog.showOpenDialog({
        properties: ['openFile', 'multiSelections']
    }, function (files) {
        if (files !== undefined) {
            // handle files
        }
    });
}); */



function handleFileSelect (e) {
    var files = e.target.files;
    if (files.length < 1) {
        return;
    }
    var file = files[0];
    var reader = new FileReader();
    /* reader.onload = onFileLoaded; */
    /* reader.readAsDataURL(file); */
	reader.addEventListener("load", e => {
        /* console.log(e.target.result, JSON.parse(reader.result)) */
        var dataFromChunkG = JSON.parse(reader.result);
        var numberOfNodes = 0, numberOfServices = 0, numberOfPods = 0,
            numberOfExternalIPs = 0, nubmerOfWorkers = 0, numberOfManagement = 0;

        numberOfNodes = dataFromChunkG.nodes.length;
        for(let i = 0; i < numberOfNodes; i++) {
            switch(dataFromChunkG.nodes[i].typ) {
                case 'SVC':
                    numberOfServices ++;
                    break;
                case 'POD':
                    numberOfPods ++;
                    break;
                case 'EXT':
                    numberOfExternalIPs ++;
                    break;
                case 'WHOST':
                    nubmerOfWorkers ++;
                    break;
                case 'MHOST':
                    numberOfManagement ++;
                    break;
            }
        }
        document.getElementById('nodes').innerHTML = numberOfNodes;
        document.getElementById('servs').innerHTML = numberOfServices;
        document.getElementById('pods').innerHTML = numberOfPods;
        document.getElementById('ips').innerHTML = numberOfExternalIPs;
        document.getElementById('workers').innerHTML = nubmerOfWorkers;
        document.getElementById('man').innerHTML = numberOfManagement;
	});	
	reader.readAsText(file);
	
	var fullPath = document.getElementById('file-input').value;
	if (fullPath) {
		var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
		var filename = fullPath.substring(startIndex);
		if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) {
			filename = filename.substring(1);
		}
        document.getElementById("path").innerHTML = "$Home/"+filename;
    }
}

/* function onFileLoaded (e) {
var match = /^data:(.*);base64,(.*)$/.exec(e.target.result);
    if (match == null) {
        throw 'Could not parse result'; // should not happen
    }
    var mimeType = match[1];
    var content = match[2];
	if (mimeType === "application/json") {
		alert(mimeType);
		alert(content);
	}
} */ 

/* $(function () {
    $('#import-pfx-button').click(function(e) {
        $('#file-input').click();
    });
    $('#file-input').change(handleFileSelect);
}); */

$(function () {
	
    $('#file-input').click();

    $('#file-input').change(handleFileSelect);
});