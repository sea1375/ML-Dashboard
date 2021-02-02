anychart.onDocumentReady(function() {
    anychart.data.loadJsonFile("https://static.anychart.com/git-storage/word-press/data/network-graph-tutorial/data.json", function(data) {
        // create chart from loaded data
        console.log(data)
        var chart = anychart.graph(data);
        // set title
        // chart.title("Network Graph showing the battles in Game of Thrones");
        // access nodes
        var nodes = chart.nodes();

        // set the size of nodes
        nodes.normal().height(5);
        // nodes.hovered().height(45);
        // nodes.selected().height(45);

        // set the fill of nodes
        nodes.normal().fill("#ffa000");
        // nodes.hovered().fill("white");
        // nodes.selected().fill("#ffa000");

        // set the stroke of nodes
        nodes.normal().stroke(null);
        // nodes.hovered().stroke("#333333", 3);
        // nodes.selected().stroke("#333333", 3);
        // draw chart
        chart.container("wait1").draw();
    });
});