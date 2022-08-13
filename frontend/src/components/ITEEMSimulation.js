// Import all necessary libraries/modules.
import React, { Component } from 'react';
import { MapContainer, TileLayer, GeoJSON, Marker, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import geoData from '../assets/Watershed.json'
import reachData from '../assets/Reach.json'

import L, { map } from 'leaflet'
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

import { CircularProgress, IconButton, Select, Snackbar, FormControl, MenuItem, InputLabel, InputAdornment, Accordion, AccordionSummary, AccordionDetails, Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, TextField, Box, Grid } from '@material-ui/core';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import EditIcon from '@material-ui/icons/Edit';
import SaveIcon from '@material-ui/icons/Save';
import KeyboardBackspaceIcon from '@material-ui/icons/ArrowBackIos';

import MuiAlert from '@material-ui/lab/Alert';

import LegendLeaflet from './Legend'
import baseline_json from '../assets/data.json'

import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import LinePlots from './LinePlots'
import SankeyPlots from './SankeyPlots';
import RadarPlots from './RadarPlots';
import HighchartsMore from 'highcharts/highcharts-more'

// Highcharts extension requirements
HighchartsMore(Highcharts) 
require("highcharts/modules/exporting")(Highcharts);
require("highcharts/modules/export-data")(Highcharts);
require("highcharts/modules/sankey")(Highcharts);



// Parameter types
let wwt_params_string = ['AS', 'ASCP', 'EBPR_basic', 'EBPR_acetate', 'EBPR_StR']

// Year month array for finding correspondence when processing data.
let master_year_month_array = ["2003-1", "2003-2", "2003-3", "2003-4", "2003-5", "2003-6", "2003-7", "2003-8", "2003-9", "2003-10", "2003-11", "2003-12", "2004-1", "2004-2", "2004-3", "2004-4", "2004-5", "2004-6", "2004-7", "2004-8", "2004-9", "2004-10", "2004-11", "2004-12", "2005-1", "2005-2", "2005-3", "2005-4", "2005-5", "2005-6", "2005-7", "2005-8", "2005-9", "2005-10", "2005-11", "2005-12", "2006-1", "2006-2", "2006-3", "2006-4", "2006-5", "2006-6", "2006-7", "2006-8", "2006-9", "2006-10", "2006-11", "2006-12", "2007-1", "2007-2", "2007-3", "2007-4", "2007-5", "2007-6", "2007-7", "2007-8", "2007-9", "2007-10", "2007-11", "2007-12", "2008-1", "2008-2", "2008-3", "2008-4", "2008-5", "2008-6", "2008-7", "2008-8", "2008-9", "2008-10", "2008-11", "2008-12", "2009-1", "2009-2", "2009-3", "2009-4", "2009-5", "2009-6", "2009-7", "2009-8", "2009-9", "2009-10", "2009-11", "2009-12", "2010-1", "2010-2", "2010-3", "2010-4", "2010-5", "2010-6", "2010-7", "2010-8", "2010-9", "2010-10", "2010-11", "2010-12", "2011-1", "2011-2", "2011-3", "2011-4", "2011-5", "2011-6", "2011-7", "2011-8", "2011-9", "2011-10", "2011-11", "2011-12", "2012-1", "2012-2", "2012-3", "2012-4", "2012-5", "2012-6", "2012-7", "2012-8", "2012-9", "2012-10", "2012-11", "2012-12", "2013-1", "2013-2", "2013-3", "2013-4", "2013-5", "2013-6", "2013-7", "2013-8", "2013-9", "2013-10", "2013-11", "2013-12", "2014-1", "2014-2", "2014-3", "2014-4", "2014-5", "2014-6", "2014-7", "2014-8", "2014-9", "2014-10", "2014-11", "2014-12", "2015-1", "2015-2", "2015-3", "2015-4", "2015-5", "2015-6", "2015-7", "2015-8", "2015-9", "2015-10", "2015-11", "2015-12", "2016-1", "2016-2", "2016-3", "2016-4", "2016-5", "2016-6", "2016-7", "2016-8", "2016-9", "2016-10", "2016-11", "2016-12", "2017-1", "2017-2", "2017-3", "2017-4", "2017-5", "2017-6", "2017-7", "2017-8", "2017-9", "2017-10", "2017-11", "2017-12", "2018-1", "2018-2", "2018-3", "2018-4", "2018-5", "2018-6", "2018-7", "2018-8", "2018-9", "2018-10", "2018-11", "2018-12"]


// Icon for showing plants on map.
let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [24, 36],
    iconAnchor: [12, 36],
});

L.Marker.prototype.options.icon = DefaultIcon;


// Color scaling function for gradient color map.

function getColor(d, unitSize, minValue) {
    // console.log(d, unitSize, minValue);
    return d > (unitSize * 8) + minValue ? '#800026' :
        d > (unitSize * 7) + minValue ? '#BD0026' :
            d > (unitSize * 6) + minValue ? '#E31A1C' :
                d > (unitSize * 5) + minValue ? '#FC4E2A' :
                    d > (unitSize * 4) + minValue ? '#FD8D3C' :
                        d > (unitSize * 2) + minValue ? '#FEB24C' :
                            d > unitSize + minValue ? '#FED976' :
                                '#FFEDA0';
}

// Alert pop up for confirming custom parameters.
function Alert(props) {
    return <MuiAlert elevation={6} variant="filled" {...props} />;
}

// Create default row format for BMP table.
function createData(bmp, coverCrop, fertilizerRate, waterways, landAlloc) {
    return { bmp, coverCrop, fertilizerRate, waterways, landAlloc };
}


// Create default row format for General table.
function createDefaultData(gep, baseline, units) {
    return { gep, baseline, units };
}

const defaultRows = [
    createDefaultData('Corn Market Price', '0.152', '$/kg'),
    createDefaultData('Soybean market price', '0.356', '$/kg'),
    createDefaultData('Biomass market price', '40.0', '$/kg (dry basis)'),
    createDefaultData('Willingness to pay', '0.95', '$/(household*year) for 1% water quality improvement'),
    createDefaultData('Interest rate', '7%', '%'),
    createDefaultData('Electricity price', '0.0638', '$/kWh'),
    createDefaultData('Natural gas price', '5.25', '$/cbf'),
];

// Create default row format for Plant table.
function createPlantData(parameter, value, unit, source) {
    return { parameter, value, unit, source }
}


export default class ITEEMSimulation extends Component {
    constructor(props) {
        super(props);
        this.handleSubwatershedTextfield = this.handleSubwatershedTextfield.bind(this);
        // this.lineData = this.lineData.bind(this);
        this.state = {
            currentLocation: { lat: 40.0978, lng: -88.8334 },
            zoom: 9.3,
            selectedWatershed: 33,
            expanded_shed: true,
            selectedPlant: '/',
            expanded_plant: true,
            menu_value: 1,
            expanded_general: true,
            baseline: 100,
            bmp1: 0,
            bmp2: 0,
            bmp3: 0,
            bmp4: 0,
            bmp5: 0,
            bmp6: 0,
            currentMatrix: new Array(45).fill(0).map(
                () => {
                    let a = new Array(6).fill(0);
                    a.unshift(100);
                    return a;
                }),
            disabled: true,
            snackOpen: false,
            snackOpen2: false,
            general_corn_market_price: 0.152,
            general_soybean_market_price: 0.356,
            general_biomass_market_price: 40,
            general_willingness_to_pay: 0.95,
            general_interst_rate: 7,
            general_electricity_price: 0.0638,
            general_natural_gas_price: 5.25,
            disabled_general: true,
            general_baseline_values: [0.152, 0.356, 40, 0.95, 7, 0.0638, 5.25],
            wwt_ifi: 0,
            wwt_ipi: 0,
            wwt_ini: 0,
            wwt_rs: 0.5,
            wwt_fc: 1.49,
            wwt_mc: 0.153,
            wwt_sh: 0.37,
            wwt_ir: 7,
            wwt_ep: 0.0638,
            wwt_ng: 5.25,

            nwwt_mporp: 351.5,
            nwwt_mpoc: 98.7,
            nwwt_mpod: 152.8,
            nwwt_ir: 7,
            nwwt_ep: 0.07,
            nwwt_ngp: 2.77,

            wwt_param: 1,
            nwwt_param: 1,

            wwt_matrix: [0, 0, 0, 0.5, 1.49, 0.153, 0.37, 7, 0.0638, 5.25],
            nwwt_matrix: [351.5, 98.7, 152.8, 7, 0.07, 2.77],
            simulated: false,

            nwwt_param_wmp1: 1,
            nwwt_param_wmp2: 1,
            nwwt_param_dmp: 1,

            returned_Json: null,
            master_data: [],
            dataReceived: false,

            unitSize: 0,
            minValue: 0,
            map: null,
            legendTitle: 'Nitrate yield (kg/ha/yr)',
            yieldMenuValue: 1,
            legend: null,

            selectedWatershedOutput: 33,

            // Default input formats for various charts, to be filled later with returned data
            lineChartNitrateOptions: {
                chart: {
                    type: 'spline'
                },
                title: {
                    text: 'My chart'
                },
                series: [
                    {
                        data: [1, 2, 1, 4, 3, 6]
                    }
                ]
            },

            lineChartPhosphorusOptions: {
                chart: {
                    type: 'spline'
                },
                title: {
                    text: 'My chart'
                },
                series: [
                    {
                        data: [1, 2, 1, 4, 3, 6]
                    }
                ]
            },
            lineChartStreamflowOptions: {
                chart: {
                    type: 'spline'
                },
                title: {
                    text: 'My chart'
                },
                series: [
                    {
                        data: [1, 2, 1, 4, 3, 6]
                    }
                ]
            },
            lineChartSedimentOptions: {
                chart: {
                    type: 'spline'
                },
                title: {
                    text: 'My chart'
                },
                series: [
                    {
                        data: [1, 2, 1, 4, 3, 6]
                    }
                ]
            },
            SankeyPlotOptions: {
                title: {
                    text: 'Highcharts Sankey Diagram'
                },
                accessibility: {
                    point: {
                        valueDescriptionFormat: '{index}. {point.from} to {point.to}, {point.weight}.'
                    }
                },
                series: [{
                    type: 'sankey',
                    keys: ['from', 'to', 'weight'],
                    data: [
                        ['Brazil', 'Portugal', 5.3333],
                        ['Brazil', 'France', 1444.2],
                    ]
                }]
            },

            RadarPlotOptions: {
                chart: {
                    polar: true
                },
                title: {
                    text: 'Highcharts Polar Chart'
                },

                subtitle: {
                    text: 'Also known as Radar Chart'
                },

                pane: {
                    startAngle: 0,
                    endAngle: 360
                },

                xAxis: {
                    tickInterval: 45,
                    min: 0,
                    max: 360,
                    labels: {
                        format: '{value}°'
                    }
                },

                yAxis: {
                    min: 0
                },

                plotOptions: {
                    series: {
                        pointStart: 0,
                        pointInterval: 45
                    },
                    column: {
                        pointPadding: 0,
                        groupPadding: 0
                    }
                },

                series: [
                {
                    type: 'line',
                    name: 'Line',
                    data: [1, 2, 3, 4, 5, 6, 7, 8]
                }, {
                    type: 'area',
                    name: 'Area',
                    data: [1, 8, 2, 7, 3, 6, 4, 5]
                }]
            }

        }

    }

    // On failure
    handleClickSnack1 = () => {
        this.setState({ snackOpen: true });
    };

    handleCloseSnack1 = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }

        this.setState({ snackOpen: false });
    };

    // On success
    handleClickSnack2 = () => {
        this.setState({ snackOpen2: true });
    };

    handleCloseSnack2 = (event, reason) => {
        if (reason === 'clickaway') {
            return;
        }

        this.setState({ snackOpen2: false });
    };

    // Set up logic when interacting with subwatershed layers.
    onEachSubWatershed = (subWatershed, layer) => {
        const id_ = subWatershed.properties.OBJECTID;

        if (this.state.dataReceived === false) {
            layer.bindTooltip("Sub-watershed " + id_.toString());
        } else {
            let arr = ['nitrate', 'phosphorus', 'streamflow'];
            let unitArr = ['kg/ha/yr', 'kg/ha/yr', 'mm/yr']

            layer.bindTooltip("Yield Data: " + this.state.returned_Json['yieldData'][arr[this.state.yieldMenuValue - 1]][id_ - 1].toFixed(3) + ' ' + unitArr[this.state.yieldMenuValue - 1]);
        }

        layer.on({
            click: (event) => {
                if (this.state.dataReceived === false) {
                    const index_ = id_.toString();
                    this.setState({
                        selectedWatershed: index_,
                        selectedPlant: '/',
                        expanded_general: false,
                        disabled: true,
                        baseline: this.state.currentMatrix[id_ - 1][0],
                        bmp1: this.state.currentMatrix[id_ - 1][1],
                        bmp2: this.state.currentMatrix[id_ - 1][2],
                        bmp3: this.state.currentMatrix[id_ - 1][3],
                        bmp4: this.state.currentMatrix[id_ - 1][4],
                        bmp5: this.state.currentMatrix[id_ - 1][5],
                        bmp6: this.state.currentMatrix[id_ - 1][6],
                    })
                    console.log(this.state.selectedWatershed);
                } else {
                    this.setState({
                        selectedWatershedOutput: id_ - 1,
                    })
                    this.lineDataNitrate("Nitrate Load Plot", 'nitrate', "Date", "Nitrate Load", this.state.selectedWatershedOutput);
                    this.lineDataPhosphorus("Phosphorus Load Plot", 'phosphorus', "Date", "Phosphorus Load", this.state.selectedWatershedOutput);
                    this.lineDataStreamflow("Streamflow Load Plot", 'streamflow', "Date", "Streamflow Load", this.state.selectedWatershedOutput);

                }
            }
        })

    };



    // Handle logics for the chart value(s) change, user clicks and etc.

    handleChange = (event) => {
        this.setState({ menu_value: event.target.value });
    };

    handleYieldMenuChange = async (event) => {
        this.setState({
            yieldMenuValue: event.target.value,
        },
            () => {
                let gradNumber = 8;
                let arr = ['nitrate', 'phosphorus', 'streamflow'];
                let diff = Math.max.apply(Math, this.state.returned_Json['yieldData'][arr[event.target.value - 1]]) - Math.min.apply(Math, this.state.returned_Json['yieldData'][arr[event.target.value - 1]])

                let minvalue = Math.min.apply(Math, this.state.returned_Json['yieldData'][arr[event.target.value - 1]])

                let legendTitleArr = ['Nitrate yield (kg/ha/yr)', 'Phosphorus yield (kg/ha/yr)', 'Streamflow load (mm/yr)'];

                this.setState({
                    minValue: minvalue,
                    unitSize: diff / gradNumber,
                    legendTitle: legendTitleArr[event.target.value - 1],
                }, () => {
                    this.state.map.removeControl(this.state.legend);
                });
            }
        )
    };

    handleSubwatershedTextfield = (event) => {
        const { target: { name, value } } = event;
        this.setState({ [name]: value });
    };


    onEditLandAllocClicked = () => {
        this.setState({
            disabled: false,
        })
        console.log('edit land alloc clicked');
    }

    onSaveLandAllocClicked = () => {
        let watershed = parseInt(this.state.selectedWatershed) - 1;
        let sum = parseInt(this.state.baseline) + parseInt(this.state.bmp1) + parseInt(this.state.bmp2) + parseInt(this.state.bmp3) + parseInt(this.state.bmp4) + parseInt(this.state.bmp5) + parseInt(this.state.bmp6);
        let params = [parseInt(this.state.baseline), parseInt(this.state.bmp1), parseInt(this.state.bmp2), parseInt(this.state.bmp3), parseInt(this.state.bmp4), parseInt(this.state.bmp5), parseInt(this.state.bmp6)]

        if (sum != 100 && sum != 0) {
            this.setState({ snackOpen: true });
            return;
        }
        this.setState(({ currentMatrix }) => ({
            currentMatrix:
                currentMatrix.map((row, i) => {
                    if (i === watershed) {
                        return params;
                    }
                    return row;
                }),
        }));

        this.setState({
            disabled: true,
            snackOpen2: true,
        })
        console.log('save land alloc clicked');
    }

    // Style for subwatershed layer, before simulation.
    style = (feature) => {
        const id_ = feature.properties.OBJECTID;
        return {
            weight: 2,
            opacity: 1,
            color: 'grey',
            dashArray: '3',
            fillOpacity: 0.3
        };
    }

    // Style for subwatershed layer, after simulation, color gradient-ed.
    style_output_mode = (feature) => {
        const id_ = feature.properties.OBJECTID;
        let arr = ['nitrate', 'phosphorus', 'streamflow']
        return {
            weight: 2,
            opacity: 1,
            fillColor: getColor(this.state.returned_Json['yieldData'][arr[this.state.yieldMenuValue - 1]][id_ - 1], this.state.unitSize, this.state.minValue),
            dashArray: '3',
            fillOpacity: 0.6,
            color: 'white',
        };
    }

    onEditGeneralBaselineClicked = () => {
        this.setState({
            disabled_general: false,
        })
        console.log('edit general baseline clicked');
    }

    onSaveGeneralBaselineClicked = () => {

        let params = [parseFloat(this.state.general_corn_market_price),
        parseFloat(this.state.general_soybean_market_price),
        parseFloat(this.state.general_biomass_market_price),
        parseFloat(this.state.general_willingness_to_pay),
        parseFloat(this.state.general_interst_rate),
        parseFloat(this.state.general_electricity_price),
        parseFloat(this.state.general_natural_gas_price)]


        this.setState({
            general_baseline_values: params,
            disabled_general: true,
            snackOpen2: true,
        })
        console.log('save general baseline clicked');
    }

    // Create plot data to feed into the ploting objects.
    lineDataNitrate = (title_, type_, xlabel_, ylabel_, subwatershed) => {
        console.log(this.state.returned_Json['loadDataList']);
        this.setState({
            lineChartNitrateOptions: {
                chart: {
                    type: 'line'
                },
                title: {
                    text: title_,
                },
                yAxis: {
                    title: {
                        text: ylabel_,
                    }
                },
                xAxis: {
                    categories: master_year_month_array,
                    title: {
                        text: xlabel_,
                    }
                },

                legend: {
                    layout: 'vertical',
                    align: 'right',
                    verticalAlign: 'middle'
                },
                series: [
                    this.state.returned_Json['loadDataList'][subwatershed][type_],
                ]
            }
        })
    }
    lineDataPhosphorus = (title_, type_, xlabel_, ylabel_, subwatershed) => {
        console.log(this.state.returned_Json['loadDataList']);
        this.setState({
            lineChartPhosphorusOptions: {
                chart: {
                    type: 'line'
                },
                title: {
                    text: title_,
                },
                yAxis: {
                    title: {
                        text: ylabel_,
                    }
                },
                xAxis: {
                    categories: master_year_month_array,
                    title: {
                        text: xlabel_,
                    }
                },

                legend: {
                    layout: 'vertical',
                    align: 'right',
                    verticalAlign: 'middle'
                },
                series: [
                    this.state.returned_Json['loadDataList'][subwatershed][type_],
                ]
            }
        })
    }

    lineDataStreamflow = (title_, type_, xlabel_, ylabel_, subwatershed) => {
        console.log(this.state.returned_Json['loadDataList']);
        this.setState({
            lineChartStreamflowOptions: {
                chart: {
                    type: 'line'
                },
                title: {
                    text: title_,
                },
                yAxis: {
                    title: {
                        text: ylabel_,
                    }
                },
                xAxis: {
                    categories: master_year_month_array,
                    title: {
                        text: xlabel_,
                    }
                },

                legend: {
                    layout: 'vertical',
                    align: 'right',
                    verticalAlign: 'middle'
                },
                series: [
                    this.state.returned_Json['loadDataList'][subwatershed][type_],
                ]
            }
        })
    }

    lineDataSankey = (title_) => {
        this.setState({
            SankeyPlotOptions: {
                title: {
                    text: title_,
                },
                accessibility: {
                    point: {
                        valueDescriptionFormat: '{index}. {point.from} to {point.to}, {point.weight}.'
                    }
                },
                series: this.state.returned_Json['plist'],

            }
        })
    }


    // Start simulation 
    onStartSimulation = () => {
        console.log('start simulation');
        console.log(JSON.stringify(this.state.currentMatrix));
        this.setState({
            simulated: true,
        });

        // Sending HTTPS request to fetch data from backend, change the fetch url if in development mode.
        // This url 'https://iteembackend.web.illinois.edu/simulate' is for produce enviroment
        // Use localhost url and change package.json (and flask backend) if develop locally.
        (async () => {
            const rawResponse = await fetch('https://iteembackend.web.illinois.edu/simulate', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                mode: 'cors',
                body: JSON.stringify(
                    {
                        subwatershed: this.state.currentMatrix,
                        wwt_param: wwt_params_string[this.state.wwt_param - 1],
                        nwwt_param_wmp1: this.state.nwwt_param_wmp1,
                        nwwt_param_wmp2: this.state.nwwt_param_wmp2,
                        nwwt_param_dmp: this.state.nwwt_param_dmp,
                    })
            });

            // Catch response / error.
            let resp;
            try {
                resp = await rawResponse.json();
            } catch (e) {
                console.log(e)
            }

            // console.log(resp);
            this.setState({
                returned_Json: resp
            })

            // console.log(resp);

            this.lineDataNitrate("Nitrate Load Plot", 'nitrate', "Date", "Nitrate Load", 33);
            this.lineDataPhosphorus("Phosphorus Load Plot", 'phosphorus', "Date", "Phosphorus Load", 33);
            this.lineDataStreamflow("Streamflow Load Plot", 'streamflow', "Date", "Streamflow Load", 33);
            this.lineDataSankey("P Flow Plot");
            console.log(this.state.SankeyPlotOptions);

            let data = []
            let gradNumber = 8;
            let arr = ['nitrate', 'phosphorus', 'streamflow'];

            // Getting min value and difference to properly slice the data
            let diff = await Math.max.apply(Math, resp['yieldData'][arr[this.state.yieldMenuValue - 1]]) - Math.min.apply(Math, resp['yieldData'][arr[this.state.yieldMenuValue - 1]]);
            let minvalue = Math.min.apply(Math, resp['yieldData'][arr[this.state.yieldMenuValue - 1]]);

            this.setState({
                master_data: data,
                unitSize: diff / gradNumber,
                minValue: minvalue,
                dataReceived: true,
            })

            // console.log(data);
            // TODO: yield three diagrams - ['nitrate', 'phosphorus', 'streamflow']
            // resp['yieldData']['nitrate'] - DONE 

            // resp['yieldData']['phosphorus']
            // resp['yieldData']['streamflow']

        })();

    }

    getloadDataNitrate = (x) => {
        return x.loadDataNitrate[this.state.selectedWatershedOutput] / 1000;
    }

    getloadDataPhosphorus = (x) => {
        return x.loadDataPhosphorus[this.state.selectedWatershedOutput] / 1000;
    }

    getloadDataStreamflow = (x) => {
        return x.loadDataStreamflow[this.state.selectedWatershedOutput] / 1000000;
    }

    getsedimentLoad = (x) => {
        return x.sedimentLoad[this.state.selectedWatershedOutput] / 1000;
    }

    /// --- for baseline
    getloadDataNitrateBase = (x) => {
        return x.loadDataNitrateBase[this.state.selectedWatershedOutput] / 1000;
    }

    getloadDataPhosphorusBase = (x) => {
        return x.loadDataPhosphorusBase[this.state.selectedWatershedOutput] / 1000;
    }

    getloadDataStreamflowBase = (x) => {
        return x.loadDataStreamflowBase[this.state.selectedWatershedOutput] / 1000000;
    }

    getsedimentLoadBase = (x) => {
        return x.sedimentLoadBase[this.state.selectedWatershedOutput] / 1000;
    }


    onBackToInput = () => {
        console.log("button clicked");
        console.log(this.state.map);
        console.log(this.state.legend);

        this.state.map.removeControl(this.state.legend);
        this.setState({
            legendExist: null,
            simulated: false,
            dataReceived: false,
        })
    }



    onSavePlantInfo = () => {

        if (this.state.selectedPlant == 'Waste Water Treatment Plant') {

            let params = [
                parseFloat(this.state.wwt_ifi),
                parseFloat(this.state.wwt_ipi),
                parseFloat(this.state.wwt_ini),
                parseFloat(this.state.wwt_rs),
                parseFloat(this.state.wwt_fc),
                parseFloat(this.state.wwt_mc),
                parseFloat(this.state.wwt_sh),
                parseFloat(this.state.wwt_ir),
                parseFloat(this.state.wwt_ep),
                parseFloat(this.state.wwt_ng)
            ];

            let selection = this.state.wwt_param;

            this.setState({
                wwt_matrix: params,
                wwt_param: selection,
                snackOpen2: true,
            })



        } else {

            let params = [
                parseFloat(this.state.nwwt_mporp),
                parseFloat(this.state.nwwt_mpoc),
                parseFloat(this.state.nwwt_mpod),
                parseFloat(this.state.nwwt_ir),
                parseFloat(this.state.nwwt_ep),
                parseFloat(this.state.nwwt_ngp)
            ]
            let selection = this.state.nwwt_param;

            this.setState({
                nwwt_matrix: params,
                nwwt_param: selection,
                snackOpen2: true,
            })

            if (this.state.selectedPlant == "Plant A: Wet milling corn") {
                this.setState({
                    nwwt_param_wmp1: selection,
                })
            }

            if (this.state.selectedPlant == "Plant B: Wet milling corn") {
                this.setState({
                    nwwt_param_wmp2: selection,
                })
            }

            if (this.state.selectedPlant == "Plant C: Dry grind corn") {
                this.setState({
                    nwwt_param_dmp: selection,
                })
            }

        }


        console.log('save plant clicked');
    }



    render() {
        const
            { currentLocation,
                zoom,
                selectedWatershed,
                expanded_shed,
                selectedPlant,
                expanded_plant,
                menu_value,
                expanded_general,
                baseline,
                bmp1,
                bmp2,
                bmp3,
                bmp4,
                bmp5,
                bmp6,
                disabled,
                snackOpen,
                snackOpen2,
                general_corn_market_price,
                general_soybean_market_price,
                general_biomass_market_price,
                general_willingness_to_pay,
                general_interst_rate,
                general_electricity_price,
                general_natural_gas_price,
                disabled_general,
                wwt_ifi,
                wwt_ipi,
                wwt_ini,
                wwt_rs,
                wwt_fc,
                wwt_mc,
                wwt_sh,
                wwt_ir,
                wwt_ep,
                wwt_ng,

                nwwt_mporp,
                nwwt_mpoc,
                nwwt_mpod,
                nwwt_ir,
                nwwt_ep,
                nwwt_ngp,

                // wwt_param,
                // nwwt_param,
                simulated,
                // wwt_matrix,
                // nwwt_matrix,
                master_data,
                returned_Json,
                dataReceived,

                unitSize,
                minValue,
                map,
                legendTitle,
                yieldMenuValue,
                legend,
                selectedWatershedOutput,
                lineChartNitrateOptions,
                lineChartPhosphorusOptions,
                lineChartStreamflowOptions,
                lineChartSedimentOptions,
                SankeyPlotOptions,
                RadarPlotOptions,

            } = this.state;

        // For general information
        const defaultRows = [
            createDefaultData('Corn Market Price', <TextField disabled={disabled_general} id="outlined-basic" name='general_corn_market_price' value={general_corn_market_price} onChange={this.handleSubwatershedTextfield} variant='outlined' size='small' />, '$/kg'),
            createDefaultData('Soybean market price', <TextField disabled={disabled_general} id="outlined-basic" name='general_soybean_market_price' value={general_soybean_market_price} onChange={this.handleSubwatershedTextfield} variant='outlined' size='small' />, '$/kg'),
            createDefaultData('Biomass market price', <TextField disabled={disabled_general} id="outlined-basic" name='general_biomass_market_price' value={general_biomass_market_price} onChange={this.handleSubwatershedTextfield} variant='outlined' size='small' />, '$/kg (dry basis)'),
            createDefaultData('Willingness to pay', <TextField disabled={disabled_general} id="outlined-basic" name='general_willingness_to_pay' value={general_willingness_to_pay} onChange={this.handleSubwatershedTextfield} variant='outlined' size='small' />, '$/(household*year) for 1% water quality improvement'),
            createDefaultData('Interest rate', <TextField disabled={disabled_general} id="outlined-basic" name='general_interst_rate' value={general_interst_rate} onChange={this.handleSubwatershedTextfield} variant='outlined' size='small' />, '%'),
            createDefaultData('Electricity price', <TextField disabled={disabled_general} id="outlined-basic" name='general_electricity_price' value={general_electricity_price} onChange={this.handleSubwatershedTextfield} variant='outlined' size='small' />, '$/kWh'),
            createDefaultData('Natural gas price', <TextField disabled={disabled_general} id="outlined-basic" name='general_natural_gas_price' value={general_natural_gas_price} onChange={this.handleSubwatershedTextfield} variant='outlined' size='small' />, '$/cbf'),
        ];

        // For each subwatershed - onclick
        const rows = [
            createData('Baseline (Default)', '-', '176 kg DAP/ha', '36% of ag. land', <TextField disabled={disabled} id="outlined-basic" name='baseline' value={baseline} onChange={this.handleSubwatershedTextfield} variant='outlined' size='small' defaultValue='100' InputProps={{
                endAdornment: <InputAdornment position="end">%</InputAdornment>,
            }} />),
            createData('BMP 1', 'No cover crop (status quo)', '30% DAP reduction', 'Not applied', <TextField disabled={disabled} id="outlined-basic" name='bmp1' value={bmp1} onChange={this.handleSubwatershedTextfield} variant='outlined' size='small' defaultValue='0' InputProps={{
                endAdornment: <InputAdornment position="end">%</InputAdornment>,
            }} />),
            createData('BMP 2', 'No cover crop (status quo)', '30% DAP reduction', 'Grass waterway (GS)', <TextField disabled={disabled} id="outlined-basic" name='bmp2' value={bmp2} onChange={this.handleSubwatershedTextfield} variant='outlined' size='small' defaultValue='0' InputProps={{
                endAdornment: <InputAdornment position="end">%</InputAdornment>,
            }} />),
            createData('BMP 3', 'Winter cover crop', '30% DAP reduction', 'Not applied', <TextField disabled={disabled} id="outlined-basic" variant='outlined' name='bmp3' value={bmp3} onChange={this.handleSubwatershedTextfield} size='small' defaultValue='0' InputProps={{
                endAdornment: <InputAdornment position="end">%</InputAdornment>,
            }} />),
            createData('BMP 4', 'Winter cover crop', '30% DAP reduction', 'Filter Strips (FS)', <TextField disabled={disabled} id="outlined-basic" variant='outlined' name='bmp4' value={bmp4} onChange={this.handleSubwatershedTextfield} size='small' defaultValue='0' InputProps={{
                endAdornment: <InputAdornment position="end">%</InputAdornment>,
            }} />),
            createData('BMP 5', 'Winter cover crop', '30% DAP reduction', 'Grass waterway (GS)', <TextField disabled={disabled} id="outlined-basic" variant='outlined' name='bmp5' value={bmp5} onChange={this.handleSubwatershedTextfield} size='small' defaultValue='0' InputProps={{
                endAdornment: <InputAdornment position="end">%</InputAdornment>,
            }} />),
            createData('BMP 6 (land use change)', 'Bioenergy Crop: Switch Grass', '-', 'Not applied', <TextField disabled={disabled} id="outlined-basic" variant='outlined' name='bmp6' value={bmp6} onChange={this.handleSubwatershedTextfield} size='small' defaultValue='0' InputProps={{
                endAdornment: <InputAdornment position="end">%</InputAdornment>,
            }} />),
            createData(<p style={{ fontWeight: 'bold' }}>Sum</p>, '-', '-', '-',
                <p style={{ fontWeight: 'bold' }}>{parseInt(baseline) + parseInt(bmp1) + parseInt(bmp2) + parseInt(bmp3) + parseInt(bmp4) + parseInt(bmp5) + parseInt(bmp6)} %</p>),
        ];

        // For WWT Table

        const wwtRows = [
            createPlantData('Influent Flow Index',
                <TextField id="outlined-basic" variant='outlined' size='small' name='wwt_ifi' value={wwt_ifi} onChange={this.handleSubwatershedTextfield} defaultValue='0' InputProps={{
                    endAdornment: <InputAdornment position="end">%</InputAdornment>,
                }} />,
                '% Change from Baseline',
                '-'
            ),
            createPlantData('Influent P Index',
                <TextField id="outlined-basic" variant='outlined' size='small' name='wwt_ipi' value={wwt_ipi} onChange={this.handleSubwatershedTextfield} defaultValue='0' InputProps={{
                    endAdornment: <InputAdornment position="end">%</InputAdornment>,
                }} />,
                '% Change from Baseline',
                '-'
            ),
            createPlantData('Influent N Index',
                <TextField id="outlined-basic" variant='outlined' size='small' name='wwt_ini' value={wwt_ini} onChange={this.handleSubwatershedTextfield} defaultValue='0' InputProps={{
                    endAdornment: <InputAdornment position="end">%</InputAdornment>,
                }} />,
                '% Change from Baseline',
                '-'
            ),
            createPlantData('Recovered Struvite',
                <TextField id="outlined-basic" variant='outlined' size='small' name='wwt_rs' value={wwt_rs} onChange={this.handleSubwatershedTextfield} defaultValue='0.5' InputProps={{
                    endAdornment: <InputAdornment position="end">$</InputAdornment>,
                }} />,
                '$/Kg',
                'Assumption considering DAP market price'
            ),
            createPlantData('Ferric Chloride (40%)',
                <TextField id="outlined-basic" variant='outlined' size='small' name='wwt_fc' value={wwt_fc} onChange={this.handleSubwatershedTextfield} defaultValue='1.49' InputProps={{
                    endAdornment: <InputAdornment position="end">$</InputAdornment>,
                }} />,
                '$/Kg',
                'Bid tabulation (online)'
            ),
            createPlantData('Magnesium Chloride (30%)',
                <TextField id="outlined-basic" variant='outlined' size='small' name='wwt_mc' value={wwt_mc} onChange={this.handleSubwatershedTextfield} defaultValue='0.153' InputProps={{
                    endAdornment: <InputAdornment position="end">$</InputAdornment>,
                }} />,
                '$/Kg',
                'Bid tabulation (online)'
            ),
            createPlantData('Sodium hydroxide',
                <TextField id="outlined-basic" variant='outlined' size='small' name='wwt_sh' value={wwt_sh} onChange={this.handleSubwatershedTextfield} defaultValue='0.37' InputProps={{
                    endAdornment: <InputAdornment position="end">$</InputAdornment>,
                }} />,
                '$/Kg',
                'Bid tabulation (online)'
            ),
            createPlantData('Interst Rate',
                <TextField id="outlined-basic" variant='outlined' size='small' name='wwt_ir' value={wwt_ir} onChange={this.handleSubwatershedTextfield} defaultValue='7' InputProps={{
                    endAdornment: <InputAdornment position="end">%</InputAdornment>,
                }} />,
                '%',
                'Assumption'
            ),
            createPlantData('Electricity Price',
                <TextField id="outlined-basic" variant='outlined' size='small' name='wwt_ep' value={wwt_ep} onChange={this.handleSubwatershedTextfield} defaultValue='0.0638' InputProps={{
                    endAdornment: <InputAdornment position="end">$</InputAdornment>,
                }} />,
                '$/kWh',
                'Industrial user in Illinois, IEA'
            ),
            createPlantData('Natural gas price',
                <TextField id="outlined-basic" variant='outlined' size='small' name='wwt_ng' value={wwt_ng} onChange={this.handleSubwatershedTextfield} defaultValue='5.25' InputProps={{
                    endAdornment: <InputAdornment position="end">$</InputAdornment>,
                }} />,
                '$/cbf',
                'Industrial user in Illinois, IEA'
            ),
        ]

        // For nonWWT Table

        const nonWWTRows = [
            createPlantData('Market price of recovered P',
                <TextField id="outlined-basic" variant='outlined' size='small' name='nwwt_mporp' value={nwwt_mporp} onChange={this.handleSubwatershedTextfield} InputProps={{
                    endAdornment: <InputAdornment position="end">$</InputAdornment>,
                }} />,
                '$/ton',
                'Assumed and considering DAP market price'
            ),
            createPlantData('Market price of CGF',
                <TextField id="outlined-basic" variant='outlined' size='small' name='nwwt_mpoc' value={nwwt_mpoc} onChange={this.handleSubwatershedTextfield} InputProps={{
                    endAdornment: <InputAdornment position="end">$</InputAdornment>,
                }} />,
                '$/ton',
                '5-year average, USDA ERS'
            ),
            createPlantData('Market price of DDGS',
                <TextField id="outlined-basic" variant='outlined' size='small' name='nwwt_mpod' value={nwwt_mpod} onChange={this.handleSubwatershedTextfield} InputProps={{
                    endAdornment: <InputAdornment position="end">$</InputAdornment>,
                }} />,
                '$/ton',
                '5-year average, USDA ERS '
            ),
            createPlantData('Interst Rate',
                <TextField id="outlined-basic" variant='outlined' size='small' name='nwwt_ir' value={nwwt_ir} onChange={this.handleSubwatershedTextfield} InputProps={{
                    endAdornment: <InputAdornment position="end">%</InputAdornment>,
                }} />,
                '%',
                'Assumption'
            ),
            createPlantData('Electricity Price',
                <TextField id="outlined-basic" variant='outlined' size='small' name='nwwt_ep' value={nwwt_ep} onChange={this.handleSubwatershedTextfield} InputProps={{
                    endAdornment: <InputAdornment position="end">$/kWh</InputAdornment>,
                }} />,
                '$/kWh',
                'Industrial user in Illinois, IEA'
            ),
            createPlantData('Natural gas price',
                <TextField id="outlined-basic" variant='outlined' size='small' name='nwwt_ngp' value={nwwt_ngp} onChange={this.handleSubwatershedTextfield} InputProps={{
                    endAdornment: <InputAdornment position="end">$/MMBtu</InputAdornment>,
                }} />,
                '$/MMBtu',
                'Industrial user in Illinois, IEA'
            ),
        ]

        return (
            <div style={{ display: 'flex', flexDirection: 'row', backgroundColor: '#F5F5F5' }}>
                <div style={{ flex: 1 }}>
                    <MapContainer center={currentLocation} zoom={zoom}
                        whenCreated={map => {
                            // do whatever makes sense. I've set it to a ref
                            this.setState({
                                map: map,
                            })
                        }}>
                        <GeoJSON tag="Reach" data={reachData} />
                        
                        {/* Set up tiles for map */}
                        <TileLayer
                            url="https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png?api_key=67eb9efb-a33d-4104-b627-9456bbfb808e"
                            attribution='&copy; <a href="https://stadiamaps.com/">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors'
                        />
                        {/* <Markers venues={data.venues}/> */}


                        <GeoJSON key={dataReceived + unitSize} tag="Subwatersheds" data={geoData.features} onEachFeature={this.onEachSubWatershed} style={dataReceived ? this.style_output_mode : this.style} />

                        
                        {/* If data is received:  */}
                        {dataReceived ?

                            <LegendLeaflet key={unitSize} map={map} unitSize={unitSize} legendExist={legend} legendTitle={legendTitle} minValue={minValue} callBack={(legend) => {
                                this.setState({ legend: legend });
                            }}></LegendLeaflet>
                            :
                            null
                        }
                        {/* If data have not been received yet, make water plant markers for the map:  */}
                        {!dataReceived ?
                            <div>
                                <Marker tag="WWT Plant" position={[39.83121, -89.00160]} eventHandlers={{
                                    click: () => {
                                        this.setState({ selectedWatershed: '/', selectedPlant: 'Waste Water Treatment Plant' })
                                    }
                                }}>
                                    <Tooltip>Plant: WWT Plant</Tooltip>
                                </Marker>
                                <Marker tag="DWT plant" position={[39.82819, -88.95055]} eventHandlers={{
                                    click: () => {
                                        this.setState({ selectedWatershed: '/', selectedPlant: 'DWT Plant', expanded_general: false, })
                                    }
                                }}>
                                    <Tooltip>Plant: DWT Plant</Tooltip>
                                </Marker>
                                <Marker tag="Dairy feedlot" position={[40.28538, -88.50410]} eventHandlers={{
                                    click: () => {
                                        this.setState({ selectedWatershed: '/', selectedPlant: 'Dairy Feedlot', expanded_general: false, })
                                    }
                                }}>
                                    <Tooltip>Plant: Dairy Feedlot</Tooltip>
                                </Marker>

                                <Marker tag="Plant A: Wet milling corn" position={[39.86549, -88.88719]} eventHandlers={{
                                    click: () => {
                                        this.setState({ selectedWatershed: '/', selectedPlant: 'Plant A - Wet milling corn', expanded_general: false, })
                                    }
                                }}>
                                    <Tooltip>Plant A: Wet milling corn</Tooltip>
                                </Marker>
                                <Marker tag="Plant B: Wet milling corn" position={[39.84768, -88.92386]} eventHandlers={{
                                    click: () => {
                                        this.setState({ selectedWatershed: '/', selectedPlant: 'Plant B - Wet milling corn', expanded_general: false, })
                                    }
                                }}>
                                    <Tooltip>Plant B: Wet milling corn</Tooltip>
                                </Marker>
                                <Marker tag="Plant C: Dry grind corn" position={[40.46996, -88.39672]} eventHandlers={{
                                    click: () => {
                                        this.setState({ selectedWatershed: '/', selectedPlant: 'Plant C - Dry grind corn', expanded_general: false, })
                                    }
                                }}>
                                    <Tooltip>Plant C: Dry grind corn</Tooltip>
                                </Marker>
                            </div>
                            :
                            null
                        }

                    </MapContainer>
                </div>
                <div style={{ display: simulated ? 'none' : 'flex', flexDirection: 'column', alignItems: 'center', flex: 2, height: "calc(100vh - 85px)", overflowY: 'auto', paddingRight: 30, paddingLeft: 30 }}>

                    {/* onClick={() => {this.setState({ selectedWatershed: '/'})}} */}
                    <Accordion style={{ width: '100%', marginTop: 20 }} expanded={expanded_general} onChange={() => { this.setState({ expanded_general: !expanded_general }) }}>
                        <AccordionSummary
                            expandIcon={<ExpandMoreIcon />}
                            aria-controls="panel1a-content"
                            id="panel1a-header"
                        >
                            <p style={{ fontFamily: 'Noto Sans', fontWeight: 'bold' }}>Parameters</p>
                        </AccordionSummary>
                        <AccordionDetails>

                            {/* The following is for general information table */}
                            <TableContainer component={Paper}>
                                <Table style={{ minWidth: 300 }} size="small" aria-label="dense table">
                                    <TableHead style={{ backgroundColor: '#DA5902', fontFamily: 'Noto Sans' }}>
                                        <TableRow>
                                            <TableCell style={{ fontFamily: 'Noto Sans', color: 'white', fontWeight: 'bold' }}>General<br></br>Economic .</TableCell>
                                            <TableCell align="left" style={{ fontFamily: 'Noto Sans', color: 'white' }}>
                                                <div style={{ display: 'flex', flexDirection: 'row', alignContent: 'center' }}>
                                                    <p style={{ marginRight: 10 }}>Baseline</p>
                                                    <IconButton style={{ display: disabled_general ? 'block' : 'none' }} onClick={this.onEditGeneralBaselineClicked}>
                                                        <EditIcon />
                                                    </IconButton>
                                                    <IconButton style={{ display: disabled_general ? 'none' : 'block' }} onClick={this.onSaveGeneralBaselineClicked}>
                                                        <SaveIcon />
                                                    </IconButton>
                                                </div>
                                            </TableCell>
                                            <TableCell align="left" style={{ fontFamily: 'Noto Sans', color: 'white' }}>Unit</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {defaultRows.map((row) => (
                                            <TableRow key={row.gep}>
                                                <TableCell component="th" scope="row" style={{ fontFamily: 'Noto Sans' }}>
                                                    {row.gep}
                                                </TableCell>
                                                <TableCell align="left" style={{ fontFamily: 'Noto Sans' }}>{row.baseline}</TableCell>
                                                <TableCell align="left" style={{ fontFamily: 'Noto Sans', maxWidth: 120 }}>{row.units}</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </AccordionDetails>
                    </Accordion>
                    {/* The following is for table when user clicks a subwatershed */}
                    <Accordion style={{ display: selectedWatershed == '/' ? 'none' : 'block' }} expanded={expanded_shed} onChange={() => {
                        this.setState({ expanded_shed: !expanded_shed });
                    }}>
                        <AccordionSummary
                            expandIcon={<ExpandMoreIcon />}
                            aria-controls="panel1a-content"
                            id="panel1a-header"
                        >
                            <p style={{ fontFamily: 'Noto Sans', fontWeight: 'bold' }}>Subwatershed: #{selectedWatershed}</p>
                        </AccordionSummary>
                        <AccordionDetails>
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                                <TableContainer component={Paper}>
                                    <Table style={{ minWidth: 300 }} size="small" aria-label="simple table">
                                        <TableHead style={{ backgroundColor: '#DA5902' }}>
                                            <TableRow>
                                                <TableCell style={{ fontFamily: 'Noto Sans', color: 'white', fontWeight: 'bold' }}>BMP&nbsp;#</TableCell>
                                                <TableCell align="left" style={{ fontFamily: 'Noto Sans', color: 'white' }}>Cover Crop</TableCell>
                                                <TableCell align="left" style={{ fontFamily: 'Noto Sans', color: 'white' }}>Fertilizer Rate</TableCell>
                                                <TableCell align="left" style={{ fontFamily: 'Noto Sans', color: 'white' }}>Waterways/Buffers</TableCell>
                                                <TableCell align="left" style={{ fontFamily: 'Noto Sans', color: 'white', minWidth: 80 }}>
                                                    <div style={{ display: 'flex', flexDirection: 'row' }}>
                                                        Land<br />Allocation
                                                        {/* <div style={{width: 5}}></div> */}
                                                        <IconButton style={{ display: disabled ? 'block' : 'none' }} onClick={this.onEditLandAllocClicked}>
                                                            <EditIcon />
                                                        </IconButton>
                                                        <IconButton style={{ display: disabled ? 'none' : 'block' }} onClick={this.onSaveLandAllocClicked}>
                                                            <SaveIcon />
                                                        </IconButton>
                                                    </div>
                                                </TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {rows.map((row) => (
                                                <TableRow key={row.bmp}>
                                                    <TableCell component="th" scope="row" style={{ fontFamily: 'Noto Sans' }}>
                                                        {row.bmp}
                                                    </TableCell>
                                                    <TableCell align="left" style={{ fontFamily: 'Noto Sans' }}>{row.coverCrop}</TableCell>
                                                    <TableCell align="left" style={{ fontFamily: 'Noto Sans' }}>{row.fertilizerRate}</TableCell>
                                                    <TableCell align="left" style={{ fontFamily: 'Noto Sans' }}>{row.waterways}</TableCell>
                                                    <TableCell align="left" style={{ fontFamily: 'Noto Sans' }}>{row.landAlloc}</TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                                {/* <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', marginTop: 20, marginBottom: 20 }}>
                                    <Button variant="contained" color="primary" style={{ display: disabled ? 'block' : 'none' }} onClick={this.onEditLandAllocClicked} >
                                        Edit Land Allocation
                                    </Button>
                                    <Button variant="contained" color="secondary" style={{ display: disabled ? 'none' : 'block' }} onClick={this.onSaveLandAllocClicked} >
                                        Save Parameters
                                    </Button>
                                </div> */}
                            </div>
                        </AccordionDetails>
                    </Accordion>
                    <Accordion style={{ display: selectedPlant == '/' ? 'none' : 'block' }} expanded={expanded_plant} onChange={() => {
                        this.setState({ expanded_plant: !expanded_plant })
                    }}>
                        <AccordionSummary
                            expandIcon={<ExpandMoreIcon />}
                            aria-controls="panel1a-content"
                            id="panel1a-header"
                        >
                            <p style={{ fontFamily: 'Noto Sans', fontWeight: 'bold' }}>Plant: #{selectedPlant}</p>
                        </AccordionSummary>
                        <AccordionDetails>
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }} >
                                <FormControl style={{ minWidth: 200, marginBottom: 20 }}>
                                    <InputLabel>Decision Variable</InputLabel>

                                    {selectedPlant == 'Waste Water Treatment Plant' ?
                                        <Select
                                            value={menu_value}
                                            onChange={this.handleChange}
                                        >
                                            <MenuItem value={1}><em>Default: Activated sludge (AS)</em></MenuItem>
                                            <MenuItem value={2}>Activated sludge with chemical precipitation (ASCP)</MenuItem>
                                            <MenuItem value={3}>Enhanced biological phosphorus removal (EBPR)</MenuItem>
                                            <MenuItem value={4}>Enhanced biological phosphorus removal with acetate addition (EBPR_acetate)</MenuItem>
                                            <MenuItem value={5}>Enhanced biological phosphorus removal with struvite precipitation (EBPR_StR)</MenuItem>
                                        </Select>
                                        :
                                        <Select
                                            value={menu_value}
                                            onChange={this.handleChange}
                                        >
                                            <MenuItem value={1}><em>Default: Baseline</em></MenuItem>
                                            <MenuItem value={2}>P Recovery</MenuItem>
                                        </Select>
                                    }
                                </FormControl>

                                <TableContainer component={Paper}>
                                    <Table style={{ minWidth: 300 }} size="small" aria-label="simple table">
                                        <TableHead style={{ backgroundColor: '#DA5902' }}>
                                            <TableRow>
                                                <TableCell style={{ fontFamily: 'Noto Sans', color: 'white', fontWeight: 'bold' }}>Parameter</TableCell>
                                                <TableCell align="left" style={{ fontFamily: 'Noto Sans', color: 'white' }}>Value</TableCell>
                                                <TableCell align="left" style={{ fontFamily: 'Noto Sans', color: 'white' }}>Unit</TableCell>
                                                <TableCell align="left" style={{ fontFamily: 'Noto Sans', color: 'white' }}>Source</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        {selectedPlant == 'Waste Water Treatment Plant' ?
                                            <TableBody>
                                                {wwtRows.map((row) => (
                                                    <TableRow key={row.parameter}>
                                                        <TableCell component="th" scope="row" style={{ fontFamily: 'Noto Sans' }}>
                                                            {row.parameter}
                                                        </TableCell>
                                                        <TableCell align="left" style={{ fontFamily: 'Noto Sans' }}>{row.value}</TableCell>
                                                        <TableCell align="left" style={{ fontFamily: 'Noto Sans' }}>{row.unit}</TableCell>
                                                        <TableCell align="left" style={{ fontFamily: 'Noto Sans' }}>{row.source}</TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                            :
                                            <TableBody>
                                                {nonWWTRows.map((row) => (
                                                    <TableRow key={row.parameter}>
                                                        <TableCell component="th" scope="row" style={{ fontFamily: 'Noto Sans' }}>
                                                            {row.parameter}
                                                        </TableCell>
                                                        <TableCell align="left" style={{ fontFamily: 'Noto Sans' }}>{row.value}</TableCell>
                                                        <TableCell align="left" style={{ fontFamily: 'Noto Sans' }}>{row.unit}</TableCell>
                                                        <TableCell align="left" style={{ fontFamily: 'Noto Sans' }}>{row.source}</TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        }
                                    </Table>
                                </TableContainer>
                                <Button color='primary' style={{ marginTop: 20 }} variant="contained" onClick={this.onSavePlantInfo}>
                                    Save Settings
                                </Button>
                            </div>
                        </AccordionDetails>
                    </Accordion>
                    <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', marginTop: 20, marginBottom: 20 }}>
                        <Button variant="contained" color="primary" onClick={this.onStartSimulation} >
                            Start Simulation
                        </Button>
                    </div>
                </div>
                <div style={{ display: simulated && !dataReceived ? 'flex' : 'none', flex: 2, height: "calc(100vh - 85px)", justifyContent: 'center', alignItems: 'center' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        <CircularProgress>
                        </CircularProgress>
                        <p style={{ fontFamily: 'Noto Sans' }}>
                            Generating Simulation Result...
                        </p>
                    </div>
                </div>
                <div style={{ display: dataReceived ? 'flex' : 'none', flexDirection: 'column', alignItems: 'center', flex: 2, height: "calc(100vh - 85px)", overflowY: 'auto', paddingRight: 30, paddingLeft: 30 }}>
                    <div style={{ display: 'flex', flexDirection: 'row', marginTop: 20, marginRight: 'auto' }} >
                        <Button startIcon={<KeyboardBackspaceIcon />} onClick={this.onBackToInput}>
                            Back
                        </Button>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'row', paddingBottom: 10 }} >
                        <h3 style={{ fontFamily: 'Noto Sans' }}>
                            Result
                        </h3>
                    </div>
                    <Accordion style={{ width: '100%' }}>
                        <AccordionSummary
                            expandIcon={<ExpandMoreIcon />}
                            aria-controls="panel1a-content"
                            id="panel1a-header"
                        >
                            <p style={{ fontFamily: 'Noto Sans', fontWeight: 'bold' }}>Spacial Yield Plots</p>
                        </AccordionSummary>
                        <AccordionDetails style={{ display: 'flex', flexDirection: 'column', alignContent: 'center' }}>

                            <div>
                                <p style={{ marginBottom: 20 }}>
                                    Note: Yield data plots will be shown on the left.
                                </p>
                                <FormControl style={{ minWidth: 200 }}>
                                    <InputLabel>Decision Variable</InputLabel>
                                    <Select
                                        value={yieldMenuValue}
                                        onChange={this.handleYieldMenuChange}
                                    >
                                        <MenuItem value={1}><em>Default: Nitrate</em></MenuItem>
                                        <MenuItem value={2}>Phosphorus</MenuItem>
                                        <MenuItem value={3}>Streamflow</MenuItem>

                                    </Select>
                                </FormControl>
                            </div>

                        </AccordionDetails>
                    </Accordion>
                    <Accordion style={{ width: '100%' }}>
                        <AccordionSummary
                            expandIcon={<ExpandMoreIcon />}
                            aria-controls="panel1a-content"
                            id="panel1a-header"
                        >
                            <p style={{ fontFamily: 'Noto Sans', fontWeight: 'bold' }}>Dynamic Load Plots for {selectedWatershedOutput === 33 ? 'General' : 'Subwatershed #' + selectedWatershedOutput}</p>
                        </AccordionSummary>
                        <AccordionDetails>
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                                <p style={{ fontFamily: 'Noto Sans', marginBottom: 30, marginLeft: 20, marginRight: 20 }}> Note: You may click on subwatersheds on the left map to see plots for a specific subwatershed.</p>
                                {/* <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', flex: 1 }} > */}
                                {/* put two charts here */}
                                {/* <LinePlots options={lineChartNitrateOptions} /> 
                                     <LinePlots options={lineChartPhosphorusOptions} /> */}

                                <LinePlots highcharts={Highcharts} options={lineChartNitrateOptions} />

                                <LinePlots highcharts={Highcharts} options={lineChartPhosphorusOptions} />


                                {/* </div> */}
                                <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', flex: 1 }} >
                                    {/* put two charts here */}
                                </div>
                            </div>
                        </AccordionDetails>
                    </Accordion>

                    <Accordion style={{ width: '100%' }}>
                        <AccordionSummary
                            expandIcon={<ExpandMoreIcon />}
                            aria-controls="panel1a-content"
                            id="panel1a-header"
                        >
                            <p style={{ fontFamily: 'Noto Sans', fontWeight: 'bold' }}>P Flow Diagram</p>
                        </AccordionSummary>
                        <AccordionDetails style={{ display: 'flex', flexDirection: 'column', alignContent: 'center' }}>
                            <p>Sankey Diagram for P Flow</p>
                            {/* TODO: sankey charts here */}
                            <SankeyPlots highcharts={Highcharts} options={SankeyPlotOptions} />

                        </AccordionDetails>
                    </Accordion>

                    <Accordion style={{ width: '100%' }}>
                        <AccordionSummary
                            expandIcon={<ExpandMoreIcon />}
                            aria-controls="panel1a-content"
                            id="panel1a-header"
                        >
                            <p style={{ fontFamily: 'Noto Sans', fontWeight: 'bold' }}>Spider Diagram</p>
                        </AccordionSummary>
                        <AccordionDetails style={{ display: 'flex', flexDirection: 'column', alignContent: 'center' }}>
                            {/* TODO: radar charts here */}
                            <RadarPlots highcharts={Highcharts} options={RadarPlotOptions} />
                        </AccordionDetails>
                    </Accordion>

                </div>




                <Snackbar open={snackOpen} autoHideDuration={1500} onClose={this.handleCloseSnack1}>
                    <Alert onClose={this.handleCloseSnack1} severity="error">
                        Error: Parameters must sum up to 100%.
                    </Alert>
                </Snackbar>
                <Snackbar open={snackOpen2} autoHideDuration={1500} onClose={this.handleCloseSnack2}>
                    <Alert onClose={this.handleCloseSnack2} severity="success">
                        Saved Successfully!
                    </Alert>
                </Snackbar>
            </div>
        );
    }
}


// 1. Label change to Baseline v.s. Alternative Scenario
// 2. For general -> for Watershed Outlet
// 3. 
