let train_parameter = {
  file_name: 'app.graphsage.supervised_train',
  train_prefix: 'app/base/static/graphs',
  data_prefix: 'adpcicd',
  model: 'gcn',
  max_degree: 10,
  epochs: 30,
  train_chunks: true,
  train_percentage: 0.8,
  validate_batch_size: -1,
  nodes_max: 1000,
};

let open_state = false;

let visualize_state = 'none';
let animation_speed = parseInt($('#speed')[0].value, 10);
let train_state = false;

const NUMBER_OF_GRAPHS = 700;
const DATA_PREFIX = 'adpcicd-G.json';
let chunk_load_state = false;
let graphs = [];
let index = 0;

let epoch_progress = 0;
let train_result = {
  train_loss: [],
  train_f1_mic: [],
  train_f1_mac: [],
  val_loss: [],
  val_f1_mic: [],
  val_f1_mac: [],
}

let dataSet = [], chart = [], series = [],
  drawPanel = ['train_loss', 'train_f1_mic', 'train_f1_mac', 'val_loss', 'val_f1_mic', 'val_f1_mac'];

window.readChunks = async function () {
  let json_url = '/static/graphs/chunk';
  json_url += index.toString().padStart(3, '0');
  json_url += '/' + DATA_PREFIX;
  $.ajax({
    url: json_url,
    type: 'GET',
    statusCode: {
      404: function () {
        // not exist.
        chunk_load_state = true;
      }
    },
    success: function (json) {
      //code here
      graphs.push(json);
      index++;
      if (index < NUMBER_OF_GRAPHS) readChunks();
    }
  });
}

function createAnychartData(dataFromChunkG) {
  let anychartData = {
    nodes: [],
    edges: [],
  }
  for (let i = 0; i < dataFromChunkG.nodes.length; i++) {
    anychartData.nodes.push({
      id: dataFromChunkG.nodes[i].id.toString(),
      group: dataFromChunkG.nodes[i].group,
      feature: dataFromChunkG.nodes[i].feature,
      type: dataFromChunkG.nodes[i].typ,
      normal: {
        fill: dataFromChunkG.nodes[i].color.toString(),
        stroke: null,
      },
      hovered: {
        fill: 'white',
        stroke: `3 ${dataFromChunkG.nodes[i].color.toString()}`,
      },
      selected: {
        fill: dataFromChunkG.nodes[i].color.toString(),
        stroke: '3 #333333',
      }
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
    let node = anychartData.nodes[i].id,
      hasLink = false;
    for (let j = 0; j < anychartData.edges.length; j++) {
      let source = anychartData.edges[j].from,
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

function drawAnychart(anychartData, containerID) {
  document.getElementById(containerID).innerHTML = '';

  let chart = anychart.graph(anychartData);
  chart.container(containerID);
  chart.background().fill(null);
  chart.background().stroke(null);

  let nodes = chart.nodes();

  nodes.normal().shape('diamond');

  nodes.normal().height(7);
  nodes.hovered().height(10);
  nodes.selected().height(10);

  chart.tooltip().useHtml(true);
  chart.tooltip().width(150);
  chart.tooltip().positionMode('chart');
  chart.tooltip().anchor('left-top');
  chart.tooltip().position('left-top');
  chart.tooltip().format(function () {
    if (this.type == 'node') {
      let tooltip_id = '<span style=\'font-weight: bold;font-size: 16px;\'>ID: ' + this.id + '</span><br>';
      let tooltip_group = 'group: ' + this.getData('group') + '<br>';
      let tooltip_type = 'type: ' + this.getData('type') + '<br>';
      let tooltip_feature = 'feature: <br>';
      for (let index = 0; index < this.getData("feature").length; index++) {
        tooltip_feature += index + ' : ' + this.getData('feature')[index] + '<br>';
      }
      return '<div style=\'text-align: left\'>' + tooltip_id + tooltip_group + tooltip_type + tooltip_feature + '</div>';
    }
  });
  chart.draw();
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function visualize_graphs() {
  try {
    for (let i = 0; i < graphs.length; i++) {
      let dataFromChunkG = graphs[i];
      let numberOfNodes = 0,
        numberOfServices = 0,
        numberOfPods = 0,
        numberOfExternalIPs = 0,
        numberOfWorkers = 0,
        numberOfManagement = 0;

      numberOfNodes = dataFromChunkG.nodes.length;
      for (let j = 0; j < numberOfNodes; j++) {
        switch (dataFromChunkG.nodes[j].typ) {
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
            numberOfWorkers++;
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
      document.getElementById('workers').innerHTML = numberOfWorkers;
      document.getElementById('man').innerHTML = numberOfManagement;

      let anychartData = createAnychartData(dataFromChunkG);

      drawAnychart(anychartData, 'graph');

      progressBar(i, graphs.length);
      let time = 0;
      while (true) {
        if (time >= animation_speed && visualize_state == 'play') {
          break;
        }
        await sleep(1000);
        time++;
      }
    }
    visualize_state = 'none';
    document.getElementsByClassName('play')[0].style.visibility = 'hidden';
    document.getElementsByClassName('stop')[0].style.visibility = 'hidden';
    document.getElementsByClassName('start')[0].style.visibility = 'visible';
  } catch (e) {
    console.log(e);
  }
}

async function visualize() {
  switch (visualize_state) {
    case 'none':
      while (!chunk_load_state) {
        document.getElementById('json-load-alert').style.display = 'block';
        let alert_message = 'Graphs is loading...       At present, ' + index.toString() + ' graphs is loaded. Please wait...';
        document.getElementById('json-load-alert').innerHTML = alert_message;
        await sleep(500);
      }
      document.getElementById('json-load-alert').style.display = 'none';

      visualize_state = 'play';
      document.getElementsByClassName('start')[0].style.visibility = 'hidden';
      document.getElementsByClassName('play')[0].style.visibility = 'visible';

      visualize_graphs().then(r => console.log('graph'));
      break;
    case 'play':
      visualize_state = 'stop';
      document.getElementsByClassName('play')[0].style.visibility = 'hidden';
      document.getElementsByClassName('stop')[0].style.visibility = 'visible';
      break;
    case 'stop':
      visualize_state = 'play';
      document.getElementsByClassName('stop')[0].style.visibility = 'hidden';
      document.getElementsByClassName('play')[0].style.visibility = 'visible';
      break;
  }
}

function handleFileSelect(e) {

  let files = e.target.files;
  if (files.length < 1) {
    return;
  }
  let file = files[0];
  let reader = new FileReader();
  try {
    reader.addEventListener('load', e => {
      // save the project parameters
      let projectData = JSON.parse(reader.result);
      train_parameter.model = projectData.algorithm;
      train_parameter.max_degree = parseInt(projectData.max_degree);
      train_parameter.epochs = parseInt(projectData.epochs);
      train_parameter.train_chunks = projectData.mode;
      train_parameter.train_percentage = parseFloat(projectData.percentage);
      train_parameter.nodes_max = parseInt(projectData.max_nodes);
      drawCharts();
    });
    reader.readAsText(file);

    let fullPath = document.getElementById('file-input').value;
    if (fullPath) {
      let startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
      let filename = fullPath.substring(startIndex);
      if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) {
        filename = filename.substring(1);
      }
      document.getElementById('project_path').innerHTML = '$Home/' + filename;
    }
    open_state = true;
  } catch (e) {
    document.getElementById('open-json-alert').style.display = 'block';
    setTimeout(function () {
      document.getElementById('open-json-alert').style.display = 'none';
    }, 3000);
  }
}

$(function () {
  readChunks().then(() => console.log('readChunks'));
  $('#file-input').click();
  $('#file-input').change(handleFileSelect);
  $('#speed').change(changeAnimationSpeed);
});

function progressBar(progressVal, totalPercentageVal = 100) {
  let strokeVal = (4.64 * 100) / totalPercentageVal;
  let x = document.querySelector('.progress-circle-prog');
  x.style.strokeDasharray = progressVal * (strokeVal) * 2 + ' 999';
  let el = document.querySelector('.progress-text');
  let from = $('.progress-text').data('progress');
  $('.progress-text').data('progress', progressVal);
  let start = new Date().getTime();

  setTimeout(function () {
    let now = (new Date().getTime()) - start;
    let progress = now / 700;
    el.innerHTML = progressVal;
    if (progress < 1) setTimeout(arguments.callee, 10);
  }, 10);

}

function changeAnimationSpeed() {
  let value = parseInt($('#speed')[0].value, 10);
  value = isNaN(value) ? 1 : value;
  if (value < 1) value = 1;
  animation_speed = value;
}

function increaseSpeed() {
  animation_speed++;
  $('#speed')[0].value = animation_speed;
}

function decreaseSpeed() {
  if (animation_speed > 1) animation_speed--;
  $('#speed')[0].value = animation_speed;
}

function goToTrain() {
  $('#tabs-icons-text-1-tab').removeClass('active');
  $('#tabs-icons-text-2-tab').addClass('active');
}

function goToAnalysis() {
  $('#tabs-icons-text-2-tab').removeClass('active');
  $('#tabs-icons-text-3-tab').addClass('active');
}

async function train() {
  if (!open_state) {
    document.getElementById('train-alert').style.display = 'block';
    setTimeout(function () {
      document.getElementById('train-alert').style.display = 'none';
    }, 3000);
    return;
  }
  train_state = true;
  $.ajax({
    type: 'GET',
    url: '/train',
    data: {
      file_name: train_parameter.file_name,
      train_prefix: train_parameter.train_prefix,
      data_prefix: train_parameter.data_prefix,
      model: train_parameter.model.toLowerCase(),
      max_degree: train_parameter.max_degree.toString(),
      epochs: train_parameter.epochs.toString(),
      train_chunks: train_parameter.train_chunks.toString(),
      train_percentage: train_parameter.train_percentage.toString(),
      validate_batch_size: train_parameter.validate_batch_size.toString(),
      nodes_max: train_parameter.nodes_max.toString(),
    },
    success: function (data) {
      train_state = false;
      document.getElementById('train-btn').innerHTML = 'Train';
      showGraphResult();
    }
  })
  document.getElementById('train-btn').innerHTML = 'Training';

  await sleep(3000);
  epoch_progress = 0;
  readTrainResult();
}

function makeId(length) {
  let result = '';
  let characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let charactersLength = characters.length;
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * charactersLength));
  }
  return result;
}

window.readTrainResult = function () {
  if (epoch_progress >= train_parameter.epochs) {
    epoch_progress = 0;
    return;
  }
  $.ajax({
    url: '/static/assets/train_result/train_result.json?unique=' + makeId(20),
    type: 'GET',
    statusCode: {
      404: async function () {
        // await sleep(100);
        // readTrainResult();
      }
    },
    success: async function (data) {
      if (data.state == 'false') {
        return;
      }
      if (epoch_progress < data.train_loss.length) {
        let difference = data.train_loss.length - epoch_progress;
        epoch_progress += difference;
        train_result = data;
        showTrainResult(difference);
      }
      await sleep(1000);
      readTrainResult();
    }
  })
}

function showTrainResult(difference) {
  document.getElementById('train_progress').style.width =
    (epoch_progress / train_parameter.epochs * 100).toString() + '%';

  for (let i = difference; i > 0; i--) {
    for (let j = 0; j < dataSet.length; j++) {
      let view = dataSet[j].mapAs();
      view.set(
        epoch_progress - i,
        'value',
        train_result[drawPanel[j]][epoch_progress - i],
      );
    }
  }
}

function drawCharts() {
  let defaultData = [];
  for (let i = 0; i < parseInt(train_parameter.epochs); i++) {
    defaultData.push({
      x: (i + 1).toString(),
      value: null,
    });
  }

  for (let i = 0; i < drawPanel.length; i++) {

    anychart.theme('lightBlue');
    dataSet[i] = anychart.data.set(defaultData);
    chart[i] = anychart.line();
    let title = chart[i].title();
    title.enabled(true);
    title.text(drawPanel[i]);

    series[i] = chart[i].line(dataSet[i]);

    // adjust tooltip
    let tooltip = series[i].tooltip();
    tooltip.format('{%value}');

    // adjust y axis
    chart[i].yAxis().labels().enabled(false);
    chart[i].yScale().minimum(0);
    if (drawPanel[i] == 'train_loss' || drawPanel[i] == 'val_loss') {
      chart[i].yScale().maximum(5);
    } else {
      chart[i].yScale().maximum(1);
    }

    chart[i].container(drawPanel[i]).draw();
  }
}

function showGraphResult() {
  $.ajax({
    url: '/static/assets/train_result/graph_result.json?unique=' + makeId(20),
    type: 'GET',
    statusCode: {
      404: function () {
      }
    },
    success: function (data) {
      let tableRef = document.getElementById('datatable-basic').getElementsByTagName('tbody')[0];

      let row = '<tr>';
      row += '<th></th>';
      row += '<th><span class="badge badge-success">' + data.avg_loss + '</span></th>';
      row += '<th><span class="badge badge-success">' + data.avg_f1_mac + '</span></th>';
      row += '<th><span class="badge badge-success">' + data.avg_f1_mic + '</span></th>';
      row += '</tr>';
      document.getElementsByClassName('average-value')[0].innerHTML = row;

      row = '';
      for (let i = 0; i < data.graph.name.length; i++) {
        row += '<tr>';
        row += '<td>' + data.graph.name[i] + '</td>';
        row += '<td>' + data.graph.loss[i] + '</td>';
        row += '<td>' + data.graph.f1_mac[i] + '</td>';
        row += '<td>' + data.graph.f1_mic[i] + '</td>';
        row += '</tr>';
      }
      tableRef.innerHTML = row;

      $.getScript('/static/assets/vendor/datatables.net/js/jquery.dataTables.min.js', function () {
        $.getScript('/static/assets/vendor/datatables.net-bs4/js/dataTables.bootstrap4.min.js', function () {
          $.getScript('/static/assets/vendor/datatables.net-buttons/js/dataTables.buttons.min.js', function () {
            $.getScript('/static/assets/vendor/datatables.net-buttons-bs4/js/buttons.bootstrap4.min.js', function () {
              $.getScript('/static/assets/vendor/datatables.net-buttons/js/buttons.html5.min.js', function () {
                $.getScript('/static/assets/vendor/datatables.net-buttons/js/buttons.print.min.js', function () {
                  $.getScript('/static/assets/vendor/datatables.net-select/js/dataTables.select.min.js', function () {
                    console.log(document.getElementById('datatable-basic'));
                    datatableBasic();
                  });
                });
              });
            });
          });
        });
      });
    }
  })
}

function datatableBasic() {

  // Variables

  var $dtBasic = $('#datatable-basic');

  // Methods

  function init($this) {

    // Basic options. For more options check out the Datatables Docs:
    // https://datatables.net/manual/options

    var options = {
      keys: !0,
      select: {
        // style: 'multi'
      },
      language: {
        paginate: {
          previous: '<i class=\'fas fa-angle-left\'>',
          next: '<i class=\'fas fa-angle-right\'>'
        }
      },
    };

    // Init the datatable
    console.log($this);
    var table = $this.on('init.dt', function () {
      $('div.dataTables_length select').removeClass('custom-select custom-select-sm');

    }).DataTable(options);
  }

  // Events

  if ($dtBasic.length) {
    init($dtBasic);
  }
}

let tbody = document.getElementById('datatable-basic').getElementsByTagName('tbody')[0];
tbody.onclick = function (e) {
  let graphName = '';
  let target = e.target;
  while (target && target.nodeName !== 'TR') {
    target = target.parentNode;
  }
  if (target) {
    let cells = target.getElementsByTagName('td');
    graphName = cells[0].innerHTML;
    $('#analysis-tab a[href="#tabs-text-2"]').tab('show');
    document.getElementById('graph-name').innerHTML = graphName;

    for (let i = 0; i < graphs.length; i++) {
      let dataFromChunkG = graphs[i];
      if(dataFromChunkG.graph.name === graphName) {
        document.getElementById('node-number').innerHTML = dataFromChunkG.nodes.length.toString();
        drawAnychart(createAnychartData(dataFromChunkG), 'single-graph')
        break;
      }
    }
  }
};