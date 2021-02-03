var train_prefix = './graphs/',
    data_prefix = 'adpcicd',
    model = 'gcn',
    max_degree = 10,
    epochs = 30,
    train_chunks = true,
    train_percentage = 0.8,
    validata_batch_size = -1,
    nodes_max = 1000;

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

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function visualize_graphs() {

    for (let i = 0; i < graphs.length; i++) {
        var dataFromChunkG = graphs[i];
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

        document.getElementById('wait1').innerHTML = '';

        var chart = anychart.graph(anychartData);
        chart.container("wait1");
        var nodes = chart.nodes();
        nodes.normal().height(5);
        nodes.normal().fill("#ffa000");
        nodes.normal().stroke(null);

        chart.draw();

        await sleep(3000);
    }
}

function handleFileSelect(e) {

    var files = e.target.files;
    if (files.length < 1) {
        return;
    }
    var file = files[0];
    var reader = new FileReader();
    try {
        reader.addEventListener("load", e => {
            // save the project parameters
            var projectData = JSON.parse(reader.result);
            model = projectData.algorithm;
            max_degree = projectData.max_degree;
            epochs = projectData.epochs;
            train_chunks = projectData.mode;
            train_percentage = projectData.percentage;
            nodes_max = projectData.max_nodes;
            // visualize the graphs
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
        alert('visualize');
        visualize_graphs();
    } catch (e) {
        alert('Please select right json file.');
    }
}

$(function() {
    $('#file-input').click();

    $('#file-input').change(handleFileSelect);
});