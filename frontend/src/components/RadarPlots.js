import React, { Component } from 'react'
import Highcharts from 'highcharts'
import HighchartsReact from 'highcharts-react-official'
import HighchartsMore from 'highcharts/highcharts-more'
HighchartsMore(Highcharts)

require("highcharts/modules/exporting")(Highcharts);



export default class RadarPlots extends Component {
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