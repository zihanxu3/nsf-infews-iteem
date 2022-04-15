import React, { Component } from "react";

import bgImg1 from '../assets/iteemstruct1.png'
import bgImg2 from '../assets/iteemstruct2.png'

import { Divider } from '@material-ui/core';


class ITEEMStructure extends Component {
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
                        <img src={bgImg1} style={{ maxWidth: '80%' }}>
                        </img>
                        <p style={{ fontFamily: 'Noto Sans', color: 'gray' }}>
                            Figure 1. A tiered modeling framework for ITEEM.
                        </p>
                    </div>

                    <div style={{ flex: 1, marginLeft: 100, marginRight: 100, textAlign: 'justify' }}>
                        <p style={{ fontFamily: 'Noto Sans' }}>
                            Via multi-disciplinary teamwork, three process models (SWAT, WWT, and GP) and empirical (DWT) or theoretical-empirical
                            models (Economics) are first established at the process level (the lower part of Fig. 2). Then the components of
                            ITEEM are developed in the form of surrogates or empirical relationships, which are coupled by integrating input
                            and output relationships crossing temporal and spatial scales at the interaction points between the components at
                            the system level (the upper part of Fig. 2). Such a hierarchical structure allows modelers to drill down to the
                            process level and access details for better interpreting results simulated at the system level. All components of
                            the ITEEM are coded in the same programming platform, Python.
                        </p>
                    </div>
                </div>

                <Divider variant='middle' style={{ margin: 80 }} />
                <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'center', alignItems: 'center' }}>
                    <div style={{ flex: 1 }}>
                        <img src={bgImg2} style={{ maxWidth: '80%' }}>
                        </img>
                        <p style={{ fontFamily: 'Noto Sans', color: 'gray' }}>
                            Figure 2. Interaction of components in the ITEEM.
                        </p>
                    </div>

                    <div style={{ flex: 1, marginLeft: 100, marginRight: 100, marginBottom: 20, textAlign: 'justify' }}>
                        <p style={{ fontFamily: 'Noto Sans' }}>
                            Figure 2 illustrates the multiple interacting feedback loops among the components in ITEEM. SWAT has
                            the most interactions with other components. ITEEM includes five component models at the process level,
                            represented by three process-based models: 1) the Soil and Water Assessment Tool (SWAT), 2) wastewater
                            treatment model developed in GPS-X software, corn biorefinery developed in SuperPro, ) and two empirical
                            models (Economics and DWT). Detailed information about the component models with their interactions and
                            architectures of the integrated model are provided. Beyond the interactions between these components,
                            ITEEM as a whole is driven by the climate, market price of crops and rP fertilizer products, policy
                            regulations, and technology options proposed for WWT and GP components.
                        </p>

                    </div>
                </div>
            </div>
        )
    }
}

export default ITEEMStructure;
