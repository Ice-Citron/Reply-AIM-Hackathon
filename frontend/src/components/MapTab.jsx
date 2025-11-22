import { useEffect, useState, useRef } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:4000';
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;

function MapTab({ selectedHospitals }) {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const [hospitals, setHospitals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load hospitals from backend
  useEffect(() => {
    const fetchHospitals = async () => {
      try {
        const response = await fetch(`${API_URL}/api/hospitals`);
        if (!response.ok) {
          throw new Error('Failed to fetch hospitals');
        }
        const data = await response.json();
        setHospitals(data.hospitals || []);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching hospitals:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    fetchHospitals();
  }, []);

  // Initialize map
  useEffect(() => {
    console.log('MapTab: Initializing map...');
    console.log('MAPBOX_TOKEN:', MAPBOX_TOKEN ? 'Present' : 'Missing');

    if (!MAPBOX_TOKEN) {
      console.error('No Mapbox token found!');
      setError('Mapbox token not configured. Please add VITE_MAPBOX_TOKEN to your .env file.');
      setLoading(false);
      return;
    }

    if (mapRef.current) {
      console.log('Map already initialized');
      return;
    }

    if (!mapContainerRef.current) {
      console.log('Map container not ready');
      return;
    }

    console.log('Creating map with token:', MAPBOX_TOKEN.substring(0, 20) + '...');
    mapboxgl.accessToken = MAPBOX_TOKEN;

    try {
      mapRef.current = new mapboxgl.Map({
        container: mapContainerRef.current,
        style: 'mapbox://styles/mapbox/streets-v12',
        center: [0, 20],
        zoom: 2,
      });

      console.log('Map created successfully!');

      return () => {
        if (mapRef.current) {
          mapRef.current.remove();
          mapRef.current = null;
        }
      };
    } catch (err) {
      console.error('Error initializing map:', err);
      setError('Failed to initialize map. Check your Mapbox token.');
      setLoading(false);
    }
  }, []);

  // Add hospital markers
  useEffect(() => {
    const map = mapRef.current;
    if (!map || loading || hospitals.length === 0) return;

    // Clear existing markers
    if (map._hospitalMarkers) {
      map._hospitalMarkers.forEach(marker => marker.remove());
    }
    map._hospitalMarkers = [];

    // Add markers for each hospital
    hospitals.forEach(hospital => {
      const el = document.createElement('div');
      el.className = 'hospital-marker';
      el.style.width = '20px';
      el.style.height = '20px';
      el.style.borderRadius = '50%';
      el.style.backgroundColor = hospital.country === 'US' ? '#667eea' : '#f39c12';
      el.style.border = '2px solid white';
      el.style.cursor = 'pointer';
      el.style.boxShadow = '0 2px 4px rgba(0,0,0,0.3)';

      const popup = new mapboxgl.Popup({ offset: 25 }).setHTML(`
        <div class="popup-title">${hospital.name}</div>
        <div class="popup-detail">📍 ${hospital.city}, ${hospital.country}</div>
        <div class="popup-detail">⭐ Reliability: ${hospital.reliability_score}/100</div>
        <div class="popup-detail">🏥 Procedures: ${hospital.procedures.join(', ')}</div>
      `);

      const marker = new mapboxgl.Marker(el)
        .setLngLat([hospital.lng, hospital.lat])
        .setPopup(popup)
        .addTo(map);

      map._hospitalMarkers.push(marker);
    });
  }, [hospitals, loading]);

  return (
    <div className="map-container">
      {/* Separate containers for map and overlays */}
      <div className="map-wrapper">
        <div ref={mapContainerRef} style={{ width: '100%', height: '100%' }} />
        {loading && (
          <div className="loading">Loading hospitals...</div>
        )}
        {error && (
          <div className="error">
            {error}
            <br /><br />
            {!MAPBOX_TOKEN && (
              <>
                <strong>To fix this:</strong><br />
                1. Get a free token at <a href="https://mapbox.com" target="_blank" rel="noopener noreferrer">mapbox.com</a><br />
                2. Create a <code>.env</code> file in the frontend folder<br />
                3. Add: <code>VITE_MAPBOX_TOKEN=your_token_here</code><br />
                4. Restart the dev server
              </>
            )}
          </div>
        )}
      </div>
      {!loading && !error && (
        <div className="map-info">
          🔵 US Hospitals ({hospitals.filter(h => h.country === 'US').length})
          &nbsp;&nbsp;|&nbsp;&nbsp;
          🟠 International Hospitals ({hospitals.filter(h => h.country !== 'US').length})
          &nbsp;&nbsp;|&nbsp;&nbsp;
          Click markers for details
        </div>
      )}
    </div>
  );
}

export default MapTab;
