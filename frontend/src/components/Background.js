import React, { Component } from "react";

import bgImg1 from '../assets/bgImg1.png'
import bgImg2 from '../assets/bgImg2.png'

import { Divider } from '@material-ui/core';


class Background extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        // const { titleForSearch, tutorials, currentTutorial, currentIndex } = this.state;
        return (
            <div>
                <p style={{ paddingTop: 70 }}>

                </p>
                <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'center', alignItems: 'center' }}>
                    <div style={{ flex: 1 }}>
                        <img src={bgImg1} style={{ maxWidth: '90%' }}>
                        </img>
                        <p style={{ fontFamily: 'Noto Sans', color: 'gray' }}>
                            Figure 1. Risk nexus in Corn Belt
                        </p>
                    </div>

                    <div style={{ flex: 1, marginLeft: 100, marginRight: 100, textAlign: 'justify' }}>
                        <p style={{ fontFamily: 'Noto Sans' }}>
                            Food-energy-water (FEW) systems in the US Corn Belt are highly interconnected and sensitive to
                            stresses and threats (Figure 1). Grain production and subsequent utilization for animal feed,
                            human food, and ethanol production have pervasive effects on water quantity and quality in
                            downstream environments both locally (e.g., lakes and rivers with elevated nitrogen and phosphorus)
                            and nationally (e.g., Hypoxic zone in the Gulf of Mexico). Water stress associated with increased
                            climatic variability is anticipated to increase, especially in many mid-sized cities in the Corn
                            Belt that interact with neighboring agricultural lands, major industrial needs, and their shared
                            watersheds. Energy demand and overall costs for wastewater and drinking water treatment have
                            increased, and this trend is expected to be exacerbated by continued expansion of food and bioethanol production.
                        </p>

                    </div>
                </div>

                <Divider variant='middle' style={{ margin: 80 }} />
                <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'center', alignItems: 'center' }}>
                    <div style={{ flex: 1 }}>
                        <img src={bgImg2} style={{ maxWidth: '80%' }}>
                        </img>
                        <p style={{ fontFamily: 'Noto Sans', color: 'gray', marginLeft: 20 }}>
                            Figure 2. FEW systems in a Corn-Belt watershed with phosphorus (P) recovery as a key technology.
                        </p>
                    </div>

                    <div style={{ flex: 1, marginLeft: 100, marginRight: 100, marginBottom: 100, textAlign: 'justify' }}>
                        <p style={{ fontFamily: 'Noto Sans' }}>
                            Managing phosphorus (P) within FEW systems has proven especially challenging over the
                            last 40 years due to the so-called “phosphorus paradox”. On the one hand, phosphorus
                            is an essential nutrient for plant growth, and correspondingly, copious applications
                            of phosphorus fertilizer have been critical to meeting demand for food, livestock feed,
                            and biofuel. However, on the other hand, phosphorus fertilizer applied in agricultural
                            fields is at risk of being transported into water bodies where, in excess, it contributes
                            to water quality degradation, namely, toxic algae blooms. Efforts to navigate these
                            conflicting objectives have been undermined by long-lasting stores of P in fields and
                            streams (i.e. P legacy) which create time-lags between changed agricultural practices and
                            their impact on water quality or crop yields. Developing technologies for P removal and
                            recovery from waste streams with feasible costs are important given that these interventions
                            play unique and under-emphasized roles within the FEW nexus regarding water pollution,
                            resource recovery, and agriculture production, as shown in Figure 2.
                        </p>

                    </div>
                </div>
            </div>
        )
    }
}

export default Background;
