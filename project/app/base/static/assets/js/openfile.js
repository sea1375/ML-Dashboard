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
function createAnychartData(dataFromChunkG) {
    var anychartData = {
        nodes: [],
        edges: [],
    }
    for (let i = 0; i < dataFromChunkG.nodes.length; i++) {
        anychartData.nodes.push({
            id: dataFromChunkG.nodes[i].id.toString()
        });
    }
    for (let i = 0; i < dataFromChunkG.links.length; i++) {
        if (dataFromChunkG.links[i].source.toString() == dataFromChunkG.links[i].target.toString()) {
            continue;
        }
        anychartData.edges.push({
            from: dataFromChunkG.links[i].source.toString(),
            to: dataFromChunkG.links[i].target.toString()
        });
    }

    for (let i = anychartData.nodes.length - 1; i >= 0; i--) {
        var node = anychartData.nodes[i].id,
            hasLink = false;
        for (let j = 0; j < anychartData.edges.length; j++) {
            var source = anychartData.edges[j].from,
                target = anychartData.edges[j].to;
            if (node == source || node == target) {
                hasLink = true;
                break;
            }
        }
        if (!hasLink) {
            anychartData.nodes.splice(i, 1);
        }
    }
    return anychartData;
}

function handleFileSelect(e) {
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
        var numberOfNodes = 0,
            numberOfServices = 0,
            numberOfPods = 0,
            numberOfExternalIPs = 0,
            nubmerOfWorkers = 0,
            numberOfManagement = 0;

        numberOfNodes = dataFromChunkG.nodes.length;
        for (let i = 0; i < numberOfNodes; i++) {
            switch (dataFromChunkG.nodes[i].typ) {
                case 'SVC':
                    numberOfServices++;
                    break;
                case 'POD':
                    numberOfPods++;
                    break;
                case 'EXT':
                    numberOfExternalIPs++;
                    break;
                case 'WHOST':
                    nubmerOfWorkers++;
                    break;
                case 'MHOST':
                    numberOfManagement++;
                    break;
            }
        }
        document.getElementById('nodes').innerHTML = numberOfNodes;
        document.getElementById('servs').innerHTML = numberOfServices;
        document.getElementById('pods').innerHTML = numberOfPods;
        document.getElementById('ips').innerHTML = numberOfExternalIPs;
        document.getElementById('workers').innerHTML = nubmerOfWorkers;
        document.getElementById('man').innerHTML = numberOfManagement;

        var anychartData = createAnychartData(dataFromChunkG);

        // create a chart and set the data
        var chart = anychart.graph(anychartData);

        // set the container id
        chart.container("wait1");

        // initiate drawing the chart
        chart.draw();
    });
    reader.readAsText(file);

    var fullPath = document.getElementById('file-input').value;
    if (fullPath) {
        var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
        var filename = fullPath.substring(startIndex);
        if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) {
            filename = filename.substring(1);
        }
        document.getElementById("path").innerHTML = "$Home/" + filename;
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

$(function() {

    $('#file-input').click();

    $('#file-input').change(handleFileSelect);
});