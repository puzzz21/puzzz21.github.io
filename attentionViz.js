/**
 * Code ref:
 * https://github.com/jessevig/bertviz
 */

let font = 15;
let checkboxSize = 20;
let headColors = d3.scaleOrdinal(d3.schemeCategory10);
let srcKey = 0;
let srcSelect = false;
let filter = '0';
let totHeads = 16;
let layer = 0;
let heads = new Array(totHeads).fill(true)
let layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
let h = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

main();
viz();

function main() {
    if ($('#layer option:selected').text() === "") {
        for (const layer of layers) {
            $('#layer').append($("<option />").val(layer).text(layer));
        }
    }


    $('#layer').val(layer).change();

    $('#layer').on('change', function (e) {
        layer = +e.currentTarget.value;
        d3.selectAll("#heads svg").remove();
        viz();
    });

    $('#layerWord').val(layer).change();
    $('#layerWord').on('change', function (e) {
        layer = +e.currentTarget.value;
        if (srcKey !== 0) {
            selectRes(params['attention'], srcKey)
        }
    });

    $('#attentionWord').val(layer).change();
    $('#attentionWord').on('change', function (e) {
        if (srcKey !== 0) {
            selectRes(params['attention'], srcKey)
        }
    });

    $('#layerCorr').val(layer).change();
    $('#layerCorr').on('change', function (e) {
        layer = +e.currentTarget.value;
        d3.selectAll("#corr svg").remove()
        init()
    });

    $('#attentionCorr').val(layer).change();
    $('#attentionCorr').on('change', function (e) {
        d3.selectAll("#corr svg").remove()
        init()
    });

    $('#filter').on('change', function (e) {
        filter = e.currentTarget.value;
    });

}

function viz() {
    const attnData = params['attention'][filter];
    const source = attnData.src;
    const target = attnData.target;
    const layerAttention = attnData.attn[$('#layer option:selected').text()]
    $('#vis').empty();

    const svg = d3.select('#vis')
        .append('svg')
        .attr("width", "100%")
        .attr("height", 320 + "px");

    text(svg, source, true, layerAttention, 0);
    text(svg, target, false, layerAttention, 200);

    atten(svg, layerAttention);

    const checksvg = d3.select('#heads')
        .append('svg')
        .attr("width", "100%")
        .attr("height", 320 + "px");


    checkbox(0, checksvg, svg, layerAttention);
}

function text(svg, text, isSrc, attention, height) {
    var textt = d3.select("body").append("div")
        .attr("class", "tooltip").style("font-size", "15px").style("width", "auto").style("padding", "5px")

    const tokenContainer = svg.append("g").selectAll("g")
        .data(text)
        .enter()
        .append("g")
        .attr("idx", function (d, i) {
            return i
        })

    tokenContainer.append("text")
        .text((d, i) => {
            return d.replaceAll("▁", "")
        })
        .attr("font-size", font + "px")
        .style("cursor", "default")
        .attr("dx", (d, i) => i * 5 * font)
        .attr("dy", height + font)


    tokenContainer.on("mouseover", function (d, index, i) {
        let idx = parseInt(d3.select(this).attr("idx"))

        d3.select(this).style('fill', '#9FE2BF')

        svg.select("#attention")
            .selectAll("line[visibility='visible']")
            .attr("visibility", null)

        svg.select("#attention").attr("visibility", "hidden");

        if (isSrc) {
            svg.select("#attention").selectAll("line[srcId='" + idx + "']").attr("visibility", "visible");
        } else {
            svg.select("#attention").selectAll("line[targetId='" + idx + "']").attr("visibility", "visible");
        }

    });

    tokenContainer.on("mouseleave", function (d, index, i) {
        svg.select("#attention").attr("visibility", "visible");
        d3.select(this).style('fill', "black")

    });
}

function atten(svg, attention) {
    svg.select("#attention").remove()

    svg.append("g")
        .attr("id", "attention")
        .selectAll(".headAttention")
        .data(attention)
        .enter()
        .append("g")
        .classed("headAttention", true)
        .attr("head-index", (d, i) => i)
        .selectAll(".tokenAttention")
        .data(d => d)
        .enter()
        .append("g")
        .classed("tokenAttention", true)
        .attr("srcId", (d, i) => i)
        .selectAll("line")
        .data(d => d)
        .enter()
        .append("line")
        .attr("x1", function (d, i) {
            const srcIndex = +this.parentNode.getAttribute("srcId")
            return srcIndex * 5 * font + (font / 2)
        })
        .attr("y1", font)
        .attr("x2", (d, i) => i * 5 * font)
        .attr("y2", 200)
        .attr("stroke-width", 2)
        .attr("stroke", function () {
            const headIndex = +this.parentNode.parentNode.getAttribute("head-index");
            return headColors(headIndex)
        })
        .attr("srcId", function () {
            return +this.parentNode.getAttribute("srcId")
        })
        .attr("targetId", (d, i) => i)


    updateAttention(svg)
}

function updateAttention(svg) {
    svg.select("#attention")
        .selectAll("line")
        .attr("stroke-opacity", function (d) {
            const headIndex = +this.parentNode.parentNode.getAttribute("head-index");
            if (heads[headIndex]) {
                return (d / 2)
            } else {
                return 0.0;
            }
        })
}

function checkbox(top, svg, s) {
    svg.select("#attention").remove()

    const checkboxContainer = svg.append("g");

    var div = d3.select("body").append("div")
        .attr("class", "tooltip")

    const checkbox = checkboxContainer.selectAll("rect")
        .data(heads)
        .enter()
        .append("rect")
        .attr("class", "checks")
        .attr("fill", (d, i) => headColors(i))
        .attr("x", top)
        .attr("y", (d, i) => i * checkboxSize)
        .attr("idx", function (d, i) {
            return i
        })
        .attr("width", checkboxSize)
        .attr("height", checkboxSize)
        .on("mouseover", function (event, d, i) {
            let idx = parseInt((d3.select(this).attr("idx")))
            div.transition()
                .duration(200)
                .style("opacity", .9);
            div.html("Head: " + idx)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY + 10) + "px");
        })
        .on("mouseout", function (d) {
            div.transition()
                .duration(500)
                .style("opacity", 0);
        })


    function updateCheckboxes() {
        checkboxContainer.selectAll("rect")
            .data(heads)
            .attr("fill", (d, i) => d ? headColors(i) : lighten(headColors(i)));
    }

    updateCheckboxes();

    checkbox.on("click", function (d, i) {
        let idx = parseInt((d3.select(this).attr("idx")))
        if (heads[idx] && activeHeads() === 1) return;
        heads[idx] = !heads[idx];
        updateCheckboxes();
        updateAttention(s);
    });
}

function activeHeads() {
    return heads.reduce(function (acc, val) {
        return val ? acc + 1 : acc;
    }, 0);
}

function lighten(color) {
    const c = d3.hsl(color);
    const increment = (1 - c.l) * 0.6;
    c.l += increment;
    c.s -= increment;
    return c;
}
