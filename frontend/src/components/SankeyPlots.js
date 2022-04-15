import React, { Component } from 'react'
import Highcharts from 'highcharts'
import HighchartsReact from 'highcharts-react-official'


require("highcharts/modules/exporting")(Highcharts);
require("highcharts/modules/sankey")(Highcharts);


export default class SankeyPlots extends Component {
    constructor() {
        super();
        // this.chart;
    }
    componentDidMount() {
        this.chart = this.refs.chart.chart;
        console.log(this.props.options);
    }
    render() {
        return (
            <HighchartsReact
                highcharts={this.props.highcharts}
                options={this.props.options}
                ref={"chart"}
            />
        );
    }
}