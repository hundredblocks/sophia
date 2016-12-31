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


//var svg = d3.select("svg"),
//    diameter = +svg.attr("width"),
//    g = svg.append("g")
//    .attr("transform", "translate(2,2)"),
//    format = d3.format(",d");
//
//var pack = d3.pack()
//    .size([diameter - 4, diameter - 4]);
//
//var root = d3.hierarchy(data)
//  .sum(function(d) { return d.NUM_REV; })
//  .sort(function(a, b) { return b.value - a.value; });
//
//var node = g.selectAll(".node")
//.data(pack(root).descendants())
//.enter().append("g")
//  .attr("class", function(d) { return d.children ? "node" : "leaf node"; })
//  .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
//
//node.append("title")
//  .text(function(d) {
//  if(d.data.name!=="total"){
//    return d.data.GROOMED[0]
//  }
//  return d.data.name + "\n" + format(d.value); });
//
//node.append("circle")
//.style("fill", function(d) {return color_scale(d.data.POS_AVG);})
//  .attr("r", function(d) { return d.r; })
//  .on("mouseover", mouseover)
//.on("mousemove", mousemove)
//.on("mouseout", mouseout);;
//
//
//
//node.filter(function(d) { return !d.children; }).append("text")
//  .attr("dy", "0.3em")
//  .text(pickCircleDisplayText);
//
//var width = 960,
//    height = 500;
//
//var svg2 = d3.select("body").append("svg")
//    .attr("width", width)
//    .attr("height", height);
//
//var div = d3.select("body").append("div")
//    .attr("class", "tooltip")
//    .style("display", "none");
//

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

//var svg = d3.select("svg"),
//    width = +svg.attr("width"),
//    height = +svg.attr("height");
//
//var fader = function(color) { return d3.interpolateRgb(color, "#fff")(0.2); },
//    color = d3.scaleOrdinal(d3.schemeCategory20.map(fader)),
//    format = d3.format(",d");
//
//var treemap = d3.treemap()
//    .tile(d3.treemapResquarify)
//    .size([width, height])
//    .round(true)
//    .paddingInner(1);
//
//  var root = d3.hierarchy(data)
//      .eachBefore(function(d) { d.data.id = (d.parent ? d.parent.data.id + "." : "") + d.data.name; })
//      .sum(sumBySize)
//      .sort(function(a, b) { return b.height - a.height || b.value - a.value; });
//
//  treemap(root);
//
//  var cell = svg.selectAll("g")
//    .data(root.leaves())
//    .enter().append("g")
//      .attr("transform", function(d) { return "translate(" + d.x0 + "," + d.y0 + ")"; });
//
//  cell.append("rect")
//      .attr("id", function(d) { return d.data.id; })
//      .attr("width", function(d) { return d.x1 - d.x0; })
//      .attr("height", function(d) { return d.y1 - d.y0; })
//      .attr("fill", function(d) { return color(d.parent.data.id); });
//
//  cell.append("clipPath")
//      .attr("id", function(d) { return "clip-" + d.data.id; })
//    .append("use")
//      .attr("xlink:href", function(d) { return "#" + d.data.id; });
//
//  cell.append("text")
//      .attr("clip-path", function(d) { return "url(#clip-" + d.data.id + ")"; })
//    .selectAll("tspan")
//      .data(function(d) {
//      console.log(d)
////        return "A";})
//      return d.data.GROOMED[0]; })
//    .enter().append("tspan")
//      .attr("x", 4)
//      .attr("y", function(d, i) { return 13 + i * 10; })
//      .text(function(d) { return d; });
//
//  cell.append("title")
//      .text(function(d) { return d.data.id + "\n" + format(d.value); });
//
//  d3.selectAll("input")
//      .data([sumBySize, sumByCount], function(d) { return d ? d.name : this.value; })
//      .on("change", changed);
//  var timeout = d3.timeout(function() {
//    d3.select("input[value=\"sumByCount\"]")
//        .property("checked", true)
//        .dispatch("change");
//  }, 2000);
//
//  function changed(sum) {
//    timeout.stop();
//
//    treemap(root.sum(sum));
//
//    cell.transition()
//        .duration(750)
//        .attr("transform", function(d) { return "translate(" + d.x0 + "," + d.y0 + ")"; })
//      .select("rect")
//        .attr("width", function(d) { return d.x1 - d.x0; })
//        .attr("height", function(d) { return d.y1 - d.y0; });
//  }
//
//function sumByCount(d) {
//  return d.children ? 0 : 1;
//}
//
//function sumBySize(d) {
//  return d.size;
//}
