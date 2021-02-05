var graphs = [];
const NUMBER_OF_GRAPHS = 700;
const DATA_PREFIX = 'adpcicd-G.json';
var index = 0;

window.readChunks = function() {
    var json_url = '/static/graphs/chunk';
    json_url += index.toString().padStart(3, '0');
    json_url += '/' + DATA_PREFIX;

    $.ajax({
        url: json_url,
        type: "GET",
        statusCode: {
            404: function() {
                // not exist.

            }
        },
        success: function(json) {
            //code here
            graphs.push(json);
            index++;
            if (index < NUMBER_OF_GRAPHS) readChunks();
        }
    });
}
$(function() {
    readChunks();
});