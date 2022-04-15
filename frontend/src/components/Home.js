import React, { Component } from "react";

import homeImg from '../assets/homeImg.png'


class Home extends Component {
    constructor(props) {
        super(props);

    }

    render() {
        // const { titleForSearch, tutorials, currentTutorial, currentIndex } = this.state;
        return (
            <div>
                <p style={{ paddingTop: 70 }}>

                </p>
                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                    <div style={{ flex: 1 }}>
                        <img src={homeImg} style={{ maxWidth: '40%' }}>
                        </img>
                    </div>
                    <div style={{ flex: 2, marginLeft: 200, marginRight: 200, textAlign: 'justify' }}>
                        <p style={{ fontFamily: 'Noto Sans' }}>Traditional approaches usually use separate disciplinary-specific models
                        and ignore or do not fully consider the impact of the FEW nexus relations that exist at certain spatial scales.
                        Such approaches cannot capture the interconnected influence of measures taken across the interdependent systems.
                        To address this general deficit in the Corn Belt and other regions, we develop an integrated technology-environment-economics
                        modeling (ITEEM) framework to quantify environment and socio-economic impacts on FEW systems via technological
                        and policy solutions.</p>

                    </div>
                </div>
            </div>
        )

    }
}
export default Home;