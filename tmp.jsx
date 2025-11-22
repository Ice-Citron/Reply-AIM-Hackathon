import React, { useState } from 'react';
import { MapPin, Search, Loader2, AlertCircle, DollarSign } from 'lucide-react';

const ClinicFinderWithPricing = () => {
  const [location, setLocation] = useState('');
  const [diagnosis, setDiagnosis] = useState('');
  const [clinics, setClinics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingPrices, setLoadingPrices] = useState(false);
  const [error, setError] = useState('');

  const findClinics = async () => {
    if (!location.trim()) {
      setError('Please enter a location');
      return;
    }

    setLoading(true);
    setError('');
    setClinics([]);

    try {
      const specialtyTerm = diagnosis.trim() 
        ? `${diagnosis} specialist` 
        : 'medical clinic';
      
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 4000,
          messages: [
            {
              role: 'user',
              content: `Find medical clinics or healthcare facilities near "${location}"${diagnosis ? ` that can treat ${diagnosis}` : ''}. 

Search the web and return ONLY a JSON array with this exact structure (no markdown, no preamble):
[
  {
    "name": "Clinic Name",
    "address": "Full street address",
    "phone": "Phone number if available",
    "type": "Type of facility (e.g., General Practice, Urgent Care, Hospital)",
    "specialty": "Relevant specialty if applicable"
  }
]

Find at least 5-8 clinics if possible. Return ONLY the JSON array, nothing else.`
            }
          ],
          tools: [
            {
              type: "web_search_20250305",
              name: "web_search"
            }
          ]
        })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      
      let responseText = '';
      for (const block of data.content) {
        if (block.type === 'text') {
          responseText += block.text;
        }
      }

      let cleanedText = responseText
        .replace(/```json\n?/g, '')
        .replace(/```\n?/g, '')
        .trim();

      const jsonMatch = cleanedText.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        cleanedText = jsonMatch[0];
      }

      const parsedClinics = JSON.parse(cleanedText);
      
      if (Array.isArray(parsedClinics) && parsedClinics.length > 0) {
        // Initialize clinics without prices
        const clinicsWithoutPrices = parsedClinics.map(clinic => ({
          ...clinic,
          price: null,
          priceRange: null,
          currency: 'USD'
        }));
        setClinics(clinicsWithoutPrices);
        
        // Fetch prices for all clinics
        await fetchPricesForClinics(clinicsWithoutPrices);
      } else {
        setError('No clinics found. Try a different location or be more specific.');
      }

    } catch (err) {
      console.error('Error finding clinics:', err);
      setError('Failed to find clinics. Please check your location and try again.');
    } finally {
      setLoading(false);
    }
  };

  const fetchPricesForClinics = async (clinicList) => {
    setLoadingPrices(true);
    
    try {
      const treatmentType = diagnosis.trim() || 'general consultation';
      
      // Create a prompt that asks for all clinic prices at once
      const clinicNames = clinicList.map(c => c.name).join(', ');
      
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 3000,
          messages: [
            {
              role: 'user',
              content: `Find the consultation/treatment prices for ${treatmentType} at these clinics: ${clinicNames}

Search the web for actual pricing information. If you cannot find exact prices, provide realistic estimated price ranges based on the type of facility and location.

Return ONLY a JSON array matching this structure (no markdown, no preamble):
[
  {
    "name": "Exact clinic name from the list",
    "price": 150,
    "priceRange": "100-200",
    "currency": "USD",
    "notes": "Brief note about the price (e.g., 'Initial consultation', 'Estimated range', 'With insurance')"
  }
]

Return data for ALL clinics in the same order. If price info isn't available online, generate realistic estimates based on facility type and location.`
            }
          ],
          tools: [
            {
              type: "web_search_20250305",
              name: "web_search"
            }
          ]
        })
      });

      if (!response.ok) {
        throw new Error(`Pricing API error: ${response.status}`);
      }

      const data = await response.json();
      
      let responseText = '';
      for (const block of data.content) {
        if (block.type === 'text') {
          responseText += block.text;
        }
      }

      let cleanedText = responseText
        .replace(/```json\n?/g, '')
        .replace(/```\n?/g, '')
        .trim();

      const jsonMatch = cleanedText.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        cleanedText = jsonMatch[0];
      }

      const pricingData = JSON.parse(cleanedText);
      
      // Merge pricing data with clinics
      const updatedClinics = clinicList.map(clinic => {
        const priceInfo = pricingData.find(p => 
          p.name.toLowerCase().includes(clinic.name.toLowerCase()) ||
          clinic.name.toLowerCase().includes(p.name.toLowerCase())
        );
        
        if (priceInfo) {
          return {
            ...clinic,
            price: priceInfo.price,
            priceRange: priceInfo.priceRange,
            currency: priceInfo.currency || 'USD',
            priceNotes: priceInfo.notes
          };
        }
        
        // Fallback: generate synthetic price if no match found
        const basePrice = clinic.type?.toLowerCase().includes('hospital') ? 200 : 
                         clinic.type?.toLowerCase().includes('urgent') ? 150 : 100;
        return {
          ...clinic,
          price: basePrice,
          priceRange: `${basePrice - 50}-${basePrice + 50}`,
          currency: 'USD',
          priceNotes: 'Estimated price range'
        };
      });
      
      setClinics(updatedClinics);
      
    } catch (err) {
      console.error('Error fetching prices:', err);
      
      // Fallback to synthetic prices
      const clinicsWithSyntheticPrices = clinicList.map(clinic => {
        const basePrice = clinic.type?.toLowerCase().includes('hospital') ? 200 : 
                         clinic.type?.toLowerCase().includes('urgent') ? 150 : 100;
        return {
          ...clinic,
          price: basePrice,
          priceRange: `${basePrice - 50}-${basePrice + 50}`,
          currency: 'USD',
          priceNotes: 'Estimated price (pricing data unavailable)'
        };
      });
      
      setClinics(clinicsWithSyntheticPrices);
    } finally {
      setLoadingPrices(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !loading) {
      findClinics();
    }
  };

  const sortedClinics = [...clinics].sort((a, b) => {
    if (a.price && b.price) return a.price - b.price;
    return 0;
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-lg shadow-xl p-8">
          <div className="flex items-center gap-3 mb-6">
            <MapPin className="w-8 h-8 text-indigo-600" />
            <h1 className="text-3xl font-bold text-gray-800">
              Clinic Finder with Pricing
            </h1>
          </div>

          <p className="text-gray-600 mb-6">
            Find nearby medical clinics with pricing information for your treatment needs.
          </p>

          <div className="space-y-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Location *
              </label>
              <input
                type="text"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="e.g., New York, NY or 10001 or London, UK"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Diagnosis / Treatment Type (Optional)
              </label>
              <input
                type="text"
                value={diagnosis}
                onChange={(e) => setDiagnosis(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="e.g., dermatology, cardiology, urgent care"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>
          </div>

          <button
            onClick={findClinics}
            disabled={loading || loadingPrices}
            className="w-full bg-indigo-600 text-white py-3 rounded-lg font-medium hover:bg-indigo-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Searching for clinics...
              </>
            ) : loadingPrices ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Fetching prices...
              </>
            ) : (
              <>
                <Search className="w-5 h-5" />
                Find Clinics with Prices
              </>
            )}
          </button>

          {error && (
            <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <p className="text-red-800">{error}</p>
            </div>
          )}

          {sortedClinics.length > 0 && (
            <div className="mt-8">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-gray-800">
                  Found {sortedClinics.length} Clinic{sortedClinics.length !== 1 ? 's' : ''}
                </h2>
                <div className="text-sm text-gray-500">
                  Sorted by price (lowest first)
                </div>
              </div>
              
              <div className="space-y-4">
                {sortedClinics.map((clinic, index) => (
                  <div
                    key={index}
                    className="border border-gray-200 rounded-lg p-5 hover:shadow-md transition-shadow bg-gray-50"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">
                          {clinic.name}
                        </h3>
                        
                        <div className="space-y-1 text-sm text-gray-600">
                          {clinic.address && (
                            <p className="flex items-start gap-2">
                              <MapPin className="w-4 h-4 mt-0.5 flex-shrink-0" />
                              <span>{clinic.address}</span>
                            </p>
                          )}
                          
                          {clinic.phone && (
                            <p>📞 {clinic.phone}</p>
                          )}
                          
                          <div className="flex flex-wrap gap-2 mt-2">
                            {clinic.type && (
                              <span className="inline-block px-2 py-1 bg-indigo-100 text-indigo-700 rounded text-xs font-medium">
                                {clinic.type}
                              </span>
                            )}
                            
                            {clinic.specialty && (
                              <span className="inline-block px-2 py-1 bg-green-100 text-green-700 rounded text-xs font-medium">
                                {clinic.specialty}
                              </span>
                            )}
                          </div>

                          {clinic.priceNotes && (
                            <p className="text-xs text-gray-500 mt-2">
                              {clinic.priceNotes}
                            </p>
                          )}
                        </div>
                      </div>

                      <div className="flex-shrink-0 text-right">
                        {clinic.price ? (
                          <div className="bg-green-50 border border-green-200 rounded-lg px-4 py-3">
                            <div className="flex items-center gap-1 text-green-700 font-bold text-xl">
                              <DollarSign className="w-5 h-5" />
                              <span>{clinic.price}</span>
                            </div>
                            {clinic.priceRange && (
                              <div className="text-xs text-green-600 mt-1">
                                Range: ${clinic.priceRange}
                              </div>
                            )}
                            <div className="text-xs text-gray-500 mt-1">
                              {clinic.currency}
                            </div>
                          </div>
                        ) : (
                          <div className="bg-gray-100 border border-gray-200 rounded-lg px-4 py-3">
                            <div className="text-sm text-gray-500">
                              {loadingPrices ? 'Loading...' : 'Price N/A'}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-800">
                  <strong>Next Steps:</strong> Use these clinic addresses with the Google Maps API 
                  to calculate travel times. You can also compare these prices with international 
                  options to determine if medical tourism is cost-effective.
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="mt-6 text-center text-sm text-gray-600">
          <p>
            Prices are fetched from web search or estimated based on facility type and location.
            <br />Always verify pricing directly with the healthcare provider.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ClinicFinderWithPricing;
