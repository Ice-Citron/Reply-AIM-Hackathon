import { useState } from 'react';
import ChatTab from './components/ChatTab';
import MapTab from './components/MapTab';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [selectedHospitals, setSelectedHospitals] = useState([]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>🏥 CareCompass</h1>
        <p>AI-Powered Medical Cost & Reliability Comparison</p>
      </header>

      <nav className="tab-nav">
        <button
          className={`tab-button ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          💬 Ask AI Consultant
        </button>
        <button
          className={`tab-button ${activeTab === 'map' ? 'active' : ''}`}
          onClick={() => setActiveTab('map')}
        >
          🗺️ Browse Hospitals
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'chat' && (
          <ChatTab setSelectedHospitals={setSelectedHospitals} setActiveTab={setActiveTab} />
        )}
        {activeTab === 'map' && (
          <MapTab selectedHospitals={selectedHospitals} />
        )}
      </main>
    </div>
  );
}

export default App;
