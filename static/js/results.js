var color_scale= d3.scaleLinear().domain([-1, -0.25, 0.25, 1]).range(['red', 'orange', 'lightgreen', 'green']);
var div, thead, tbody, rows, thead2, tbody2, rows2, thead3, tbody3, rows3;
var tableColumns = ['NUM_REV', 'NUM_SENT', 'POS_AVG']
var columnToDisplay={
  'NUM_REV':'Reviews Mentioning This Topic (Percent of Total)',
  'NUM_SENT': 'Sentences On This Topic (Percent of Total)',
  'POS_AVG': "Average Positivity"
};
$(function() {
      $.getJSON($SCRIPT_ROOT  + '/_get_summary', {
        url: url,
      }, function(data) {
        data = JSON.parse(data.result);
        setupGraph(data);
        setupFirstTable(data);
        setupSecondTable(data[0]["GROOMED_COUNT"]);
        setupThirdTable(data[0]["CHOSEN"]);
      });
      return false;
});


function setupGraph(data){
      data = prepareData(data);
      var svg = d3.select("#graph").select("svg"),
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
      .on("click", mouseover)
    .on("mouseout", mouseout);


    node.filter(function(d) { return !d.children; })
       .append("text")
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
}

function prepareData(d){
   var a = {
        "name": "total",
        "children": d
   }
   return a
}


function setupFirstTable(data){

    var table = d3.select("#table")
        .append("table")

    thead = table.append("thead");
    tbody = table.append("tbody");
    var perc_rev = parseFloat(Math.round(data[0]["PERC_REV"] * 100) / 100).toFixed(2);
    var num_rev_str = data[0]["NUM_REV"].toString() + " ("+ perc_rev.toString()+"%)"

    var perc_sent = parseFloat(Math.round(data[0]["PERC_SENT"] * 100) / 100).toFixed(2);
    var num_sent_str = data[0]["NUM_SENT"].toString() + " ("+ perc_sent.toString()+"%)"
    var first = {
    "NUM_REV": num_rev_str,
    "NUM_SENT": num_sent_str,
//    "NUM_SENT": data[0].NUM_SENT,
    "POS_AVG": parseFloat(Math.round(data[0].POS_AVG * 100) / 100).toFixed(2)
    };

    thead.append("tr")
    .selectAll("th")
    .data(tableColumns)
    .enter()
    .append("th")
    .text(function(d) { return columnToDisplay[d]; })

  rows = tbody.selectAll("tr")
    .data([first])
    .enter()
    .append("tr");

  var cells = rows.selectAll("td")
    .data(function(row) {
        return tableColumns.map(function(column) {
            return {column: column,
            value: row[column]
            };
        });
    })
    .enter()
    .append("td")
    .html(pickCellDisplayText);

}

function setupSecondTable(wordArray){
    var table2 = d3.select("#table")
        .append("table")
    thead2 = table2.append("thead");
    tbody2 = table2.append("tbody");

    thead2.append("tr")
    .selectAll("th")
    .data(["words"])
    .enter()
    .append("th")
    .text(function(d) { return "Most Common Words (Number of Occurences)"; })

    rows2 = tbody2.selectAll("tr")
    .data([wordArray])
    .enter()
    .append("tr");

    var cells = rows2.selectAll("td")
    .data(getWordList)
    .enter()
    .append("td")
    .html(pickCellDisplayText);
}

function getWordList(row) {
        var answ = "";
        row.forEach(function(w, i){
            if(i!=0){
                answ += " - "
            }
            answ += w[0] + " (" + w[1] + ")";
        })
        return[{column: "words",
        value: answ}]
    }

function setupThirdTable(sentArray){

    var table3 = d3.select("#table")
        .append("table")

    thead3 = table3.append("thead");
    tbody3 = table3.append("tbody");

    thead3.append("tr")
    .selectAll("th")
    .data(["sentence"])
    .enter()
    .append("th")
    .text(function(d) { return "MOST RELEVANT SENTENCES"; })

  rows3 = tbody3.selectAll("tr")
    .data(sentArray)
    .enter()
    .append("tr");

  var cells = rows3.selectAll("td")
    .data(function(row, i) {
        return[{column: "sentence",
        value: row}]
    })
    .enter()
    .append("td")
    .html(pickCellDisplayText);

}

function updateFirstTable(formattedData){
    rows = tbody.selectAll("tr")
    .data([formattedData]);
    rows.enter().append("tr").merge(rows);
    rows.exit().remove();

  var cells = rows.selectAll("td")
    .data(function(row) {
        var a = tableColumns.map(function(column) {
            return {column: column,
            value: row[column]
            };
        });
        return a
    });
    cells
    .enter()
    .append("td")
    .merge(cells)
    .html(pickCellDisplayText);
    cells.exit().remove();
}

function updateSecondTable(wordArray){
    rows2 = tbody2.selectAll("tr")
    .data([wordArray]);
    rows2.enter().append("tr").merge(rows);
    rows2.exit().remove();

  var cells = rows2.selectAll("td")
    .data(getWordList);
    cells
    .enter()
    .append("td")
    .merge(cells)
    .html(pickCellDisplayText);
    cells.exit().remove();
}

function updateThirdTable(sentArray){
    rows3 = tbody3.selectAll("tr")
    .data(sentArray);

    rows3.enter().append("tr").merge(rows3);
    rows3.exit().remove();

    var cells = rows3.selectAll("td")
    .data(function(row) {
        return[{column: "sentence",
        value: row}]
    });

    cells.enter()
    .append("td")
    .merge(cells)
    .html(pickCellDisplayText);
}

function pickCellDisplayText(d, i ){
    return d.value
}


function mouseover(d, i) {
    if (d.data.name!=="total"){
        var dat = d.data;
        var perc_rev = parseFloat(Math.round(dat["PERC_REV"] * 100) / 100).toFixed(2);
        var num_rev_str = dat["NUM_REV"].toString() + " ("+ perc_rev.toString()+"%)"

        var perc_sent = parseFloat(Math.round(dat["PERC_SENT"] * 100) / 100).toFixed(2);
        var num_sent_str = dat["NUM_SENT"].toString() + " ("+ perc_sent.toString()+"%)";
        var formatted = {
        "NUM_REV": num_rev_str,
        "NUM_SENT": num_sent_str,
        "POS_AVG": parseFloat(Math.round(dat.POS_AVG * 100) / 100).toFixed(2)
        };
        updateFirstTable(formatted);
        updateSecondTable(dat["GROOMED_COUNT"]);
        updateThirdTable(dat["CHOSEN"]);
    }

}


function mouseout() {
  div.style("display", "none");
}


function pickCircleDisplayText(d){
       if(d.data.name!=="total"){
       if(d.data.GROOMED.length>0){
       if(d.data["PERC_REV"]<2){
                return "";
            }
        if(d.data["PERC_REV"]<5){
                return "...";
            }

        var strResponse = "";
        var toDisp = d.data.GROOMED.length<3 ? d.data.GROOMED.length : 3;
        var sl = d.data.GROOMED.slice(0, toDisp);
        var i = 0;
        sl.forEach(function(a){
            if(i!=0){
                strResponse += " / ";
            }
            strResponse += a;
            i+=1;
        });

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