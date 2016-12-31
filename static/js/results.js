var color_scale= d3.scaleLinear().domain([-1, -0.25, 0.25, 1]).range(['red', 'orange', 'gold', 'green']);
var div;

$(function() {
      $.getJSON($SCRIPT_ROOT + '/_get_summary', {
        url: url,
      }, function(data) {
      data = JSON.parse(data.result);
      data = prepareData(data);
      var svg = d3.select("svg"),
        diameter = +svg.attr("width"),
        g = svg.append("g")
        .attr("transform", "translate(2,2)");

    var pack = d3.pack()
        .size([diameter - 4, diameter - 4]);

    var root = d3.hierarchy(data)
      .sum(function(d) { return d.NUM_REV; })
      .sort(function(a, b) { return b.value - a.value; });

    var node = g.selectAll(".node")
    .data(pack(root).descendants())
    .enter().append("g")
      .attr("class", function(d) { return d.children ? "node" : "leaf node"; })
      .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });

    node.append("title")
      .text(function(d) {
      if(d.data.name!=="total"){
        return d.data.GROOMED[0]
      }
      return d.value; });

    node.append("circle")
    .style("fill", function(d) {return color_scale(d.data.POS_AVG);})
      .attr("r", function(d) { return d.r; })
      .on("mouseover", mouseover)
    .on("mousemove", mousemove)
    .on("mouseout", mouseout);


    node.filter(function(d) { return !d.children; }).append("text")
      .attr("dy", "0.3em")
      .text(pickCircleDisplayText);

    var width = 960,
        height = 500;

    var svg2 = d3.select("body").append("svg")
        .attr("width", width)
        .attr("height", height);

    div = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("display", "none");


      });
      return false;
});

function prepareData(d){
   var a = {
        "name": "total",
        "children": d
   }
   return a
}


function mouseover(d, i) {
  div.style("display", "inline");
}

function mousemove(d,i) {
  div
      .text(pickTooltipDisplayText(d))
      .style("left", (d3.event.pageX - 34) + "px")
      .style("top", (d3.event.pageY - 12) + "px");
}

function mouseout() {
  div.style("display", "none");
}


function pickCircleDisplayText(d){
       if(d.data.name!=="total"){
       if(d.data.GROOMED.length>0){
        var strResponse = "";
        var toDisp = d.data.GROOMED.length<3 ? d.data.GROOMED.length : 3;
        var sl = d.data.GROOMED.slice(0, toDisp);
        sl.forEach(function(a){
            strResponse += a;
            strResponse += " ";
        })
        return strResponse;
       }else {
           return "?"
       }
      }
      return d.value;
  }

function pickTooltipDisplayText(d){
       if(d.data.name!=="total"){
       if(d.data.CHOSEN.length>0){
        return d.data.CHOSEN[0];
       }else {
           return "?"
       }
      }
      return d.value;
}