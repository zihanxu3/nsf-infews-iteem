// Component credit to Sayyed Hammad Ali, original article 
// https://javascript.plainenglish.io/leaflet-map-with-react-part-i-4ef4ecbdcc1b

import L from 'leaflet';

export const VenueLocationIcon = L.icon({
  iconUrl: require('../assets/venue_location_icon.svg'),
  iconRetinaUrl: require('../assets/venue_location_icon.svg'),
  iconAnchor: null,
  shadowUrl: null,
  shadowSize: null,
  shadowAnchor: null,
  iconSize: [35, 35],
  className: 'leaflet-venue-icon'
});