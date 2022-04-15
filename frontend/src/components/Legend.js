// import { Control, withLeaflet } from "react-leaflet";
import L from "leaflet";
import { useEffect } from 'react';
import './style.css'



function getColor(d, unitSize, minValue) {
  return d > (unitSize * 8) + minValue? '#800026' :
      d > (unitSize * 7) + minValue? '#BD0026' :
          d > (unitSize * 6) + minValue? '#E31A1C' :
              d > (unitSize * 5) + minValue? '#FC4E2A' :
                  d > (unitSize * 4) + minValue? '#FD8D3C' :
                      d > (unitSize * 2) + minValue ? '#FEB24C' :
                          d > unitSize + minValue ? '#FED976' :
                              '#FFEDA0';
}

var legend;

function LegendLeaflet({ map, unitSize, legendExist, legendTitle, minValue, callBack }) {
  useEffect(() => {
    if (map) {
      console.log('called :')
      legend = L.control({ position: "bottomright" });
      callBack(legend);
      legend.onAdd = () => {
        const div = L.DomUtil.create("div", "info legend");
        const grades = [0, minValue + unitSize, unitSize * 2 + minValue, unitSize * 4 + minValue, unitSize * 5 + minValue, unitSize * 6 + minValue, unitSize * 7 + minValue, unitSize * 8 + minValue];
        let labels = [];
        let from;
        let to;

        for (let i = 0; i < grades.length; i++) {
          from = grades[i].toFixed(1);
          to = grades[i + 1];

          labels.push(
            '<i style="background:' +
            getColor(from, unitSize, minValue) +
            '"></i> ' +
            from +
            (to ? "&ndash;" + to.toFixed(1) : "+")
          );
        }
        div.innerHTML += '<p>' + legendTitle + '</p>';
        div.innerHTML += labels.join("<br>");
        return div;
      };
      legend.addTo(map);
    }
  }, [map]); //here add map
  return null;
}

export default LegendLeaflet
