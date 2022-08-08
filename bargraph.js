bar('./csv/nurse.csv', 'nurse')
function bar(path, btn){
    var ttpbar = d3.select("body").append("div")
        .attr("class", "tooltip").style("font-size", "15px").style("width", "auto").style("padding", "5px")

    if(btn === "nurse"){
        $('#'+btn).removeClass("btn-secondary")
        $('#'+btn).addClass("btn-success")
        $('#doctor').removeClass("btn-success")
        $('#doctor').addClass("btn-secondary")
    } else {
        $('#'+btn).removeClass("btn-secondary")
        $('#'+btn).addClass("btn-success")
        $('#nurse').removeClass("btn-success")
        $('#nurse').addClass("btn-secondary")
    }

    d3.selectAll("#bar svg").remove();

// append the svg object to the body of the page
    const svg = d3.select("#bar")
        .append("svg")
        .attr("width", width + margin.left + margin.right )
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform",`translate(${margin.left},${margin.top})`);

// Parse the Data
    d3.csv(path).then( function(data) {
        console.log(data)
        // List of subgroups = header of the csv files = soil condition here
        const subgroups = data.columns.slice(1)

        // List of groups = species here = value of the first column called group -> I show them on the X axis
        const groups = data.map(d => d.label)


        // Add X axis
        const x = d3.scaleBand()
            .domain(groups)
            .range([0, width])
            .padding([0.2])
        svg.append("g")
            .attr("transform", `translate(0, ${height})`)
            .call(d3.axisBottom(x).tickSize(0));

        // Add Y axis
        const y = d3.scaleLinear()
            .domain([0, 90])
            .range([ height, 0 ]);
        svg.append("g")
            .call(d3.axisLeft(y));

        // Another scale for subgroup position?
        const xSubgroup = d3.scaleBand()
            .domain(subgroups)
            .range([0, x.bandwidth()])
            .padding([0.05])

        // color palette = one color per subgroup
        const color = d3.scaleOrdinal()
            .domain(subgroups)
            .range(['#e41a1c','#377eb8'])

        // Show the bars
        let b = svg.append("g")
            .selectAll("g")
            // Enter in data = loop group per group
            .data(data)
            .join("g")
            .attr("transform", d => `translate(${x(d.label)}, 0)`)
            .selectAll("rect")
            .data(function(d) { return subgroups.map(function(key) { return {key: key, value: d[key]}; }); })
            .join("rect")
            .attr("x", d => xSubgroup(d.key))
            .attr("y", d => y(d.value))
            .attr("width", xSubgroup.bandwidth())
            .attr("height", d => height - y(d.value))
            .attr("fill", d => color(d.key))
            .on("mouseover", function (event, d, i) {
                d3.select(this).attr("opacity", 0.5)

                ttpbar.transition()
                    .duration(200)
                    .style("opacity", .9);
                ttpbar.html("<b>Value:</b> " + d.value + "<br/><b>Gender:</b> " + d.key)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY + 10) + "px");
            })
            .on("mouseout", function (d) {
                d3.select(this).attr("opacity", 1)
                ttpbar.transition()
                    .duration(500)
                    .style("opacity", 0);
            })

        svg.append("text")
            .attr("text-anchor", "end")
            .attr("transform", "rotate(-90)")
            .attr("y", -margin.left + 20)
            .attr("x", -margin.top - height/2 + 20)
            .text("scores");




    })



}

