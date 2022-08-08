if ($('#layerCorr option:selected').text() === "") {
    for (const layer of layers) {
        $('#layerCorr').append($("<option />").val(layer).text(layer));
    }
}

if ($('#attentionCorr option:selected').text() === "") {
    for (const atten of [...Array(totHeads).keys()]) {
        $('#attentionCorr').append($("<option />").val(atten).text(atten));
    }
}
init()
function init() {
    var corrtt = d3.select("body").append("div")
        .attr("class", "tooltip").style("font-size", "15px").style("width", "auto").style("padding", "5px")

    const data = [];
    const leftText = params['attention'][0].src;
    const rightText = params['attention'][0].target;
    let lt = []
    let rt = []
    let atten = $('#attentionCorr option:selected').text()
    let lay = $('#layerCorr option:selected').text()

    const marginCorr = {top: 120, right: 20, bottom: 20, left: 80},
        widthCorr = 610 - marginCorr.left - marginCorr.right,
        heightCorr = 550 - marginCorr.top - marginCorr.bottom

    const svg = d3.select("#corr")
        .append("svg")
        .attr("width", widthCorr + marginCorr.left + marginCorr.right + 50)
        .attr("height", heightCorr + marginCorr.top + marginCorr.bottom)
        .append("g")
        .attr("transform", `translate(${marginCorr.left},${marginCorr.top})`);

    [...Array(params['attention'][0]['src'].length).keys()].forEach(function (dx) {
        [...Array(params['attention'][0]['target'].length).keys()].forEach(function (dy) {
            data.push({
                x: dx,
                y: dy,
                value: params['attention'][0]['attn'][lay][atten][dx][dy]
            });
        })
    });

    var max = d3.max(data, function (d, i) {
        return d.value.toFixed(2)
    });
    var min = d3.min(data, function (d, i) {
        return d.value.toFixed(2)
    });

    for (l in leftText) {
        lt.push(leftText[l].replaceAll("▁", ""))
    }
    console.log("lt", lt)

    for (r in rightText) {

        rt.push(rightText[r].replaceAll("▁", ""))
    }
    console.log(rt)

// List of all variables and number of them
    const domain = Array.from(new Set(data.map(function (d) {
        return d.x
    })))


// Create a color scale
    const color = d3.scaleLinear()
        .domain([min, max])
        .range(["#F14978", "#02209E"]);

    const colorText = d3.scaleLinear()
        .domain([min, max])
        .range(["#2D6898", "#C2E6EA"]);

// Create a size scale for bubbles on top right. Watch out: must be a rootscale!
    const size = d3.scaleSqrt()
        .domain([0, 1])
        .range([0, 10]);

// X scale
    const x = d3.scalePoint()
        .range([0, widthCorr])
        .domain(domain)

// Y scale
    const y = d3.scalePoint()
        .range([0, heightCorr])
        .domain(domain)


    console.log(lt)

//
//
// // Create the scale
    var xx = d3.scalePoint()
        .domain(["she", "is", "a", "doctor", "and", "her", "husband", "`is`", "`a`", "nurse"])
        .range([17, 530])
// .style("opacity",0)

    svg.append("g")
        .attr("transform", "translate(-20, -30)")
        .call(d3.axisTop(xx))
        .selectAll("text")
        .attr("transform", "translate(10, -20) rotate(-90)")
        .style("text-anchor", "start")
        .style("font-size", 15)

// // Create the scale
    var yy = d3.scalePoint()
        .domain(["उनी", "डाक्टर", "हुन्", "र", "उनका", "पति", "नर्स", "`हुन्`"])
        .range([15, 340])
// .style("opacity",0)
// .style("opacity",0)


    svg.append("g")
        .attr("transform", `translate(-30, -20)`)
        .call(d3.axisLeft(yy))
        .style("font-size", 15)

// Create one 'g' element for each cell of the correlogram
    const cor = svg.selectAll(".cor")
        .data(data)
        .join("g")
        .attr("class", "cor")
        .attr("transform", function (d) {
            return `translate(${x(d.x)}, ${y(d.y)})`
        })
        .on("mouseover", function (event, d, i) {
            d3.select(this).style("stroke", "black")

            corrtt.transition()
                .duration(200)
                .style("opacity", .9);
            corrtt.html("<b>Value:</b> " + d.value)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY + 10) + "px");
        })
        .on("mouseout", function (d) {
            d3.select(this).style("stroke", "none" )
            corrtt.transition()
                .duration(500)
                .style("opacity", 0);
        })


    cor.append("circle")
        .transition().duration(1500).delay(100)
        .attr("r", function (d) {
            return size(Math.abs(d.value * 15))
        })
        .style("fill", function (d) {
            return color(d.value);
        })
        .style("opacity", 0.7)


    cor.append("text")
        .attr("y", 5)
        .text(function (d) {
            return d.value.toFixed(2);
        })
        .style("font-size", 12)
        .style("text-align", "center")
        .style("fill", function (d) {
            return colorText(d.value);
        });
}
