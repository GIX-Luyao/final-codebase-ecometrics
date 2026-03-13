import React, { useState, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';

const FilterBar = ({ filters, onFilterChange, calculatedData }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [facilities, setFacilities] = useState([]);
  const [activityTypes, setActivityTypes] = useState([]);
  const [years, setYears] = useState([]);

  // Extract unique values from data
  useEffect(() => {
    if (calculatedData && calculatedData.rawData) {
      // Extract unique facilities (LowOrg)
      const uniqueFacilities = [...new Set(
        calculatedData.rawData.map(row => row.LowOrg).filter(Boolean)
      )];
      setFacilities(uniqueFacilities);

      // Extract unique activity types (Subtype)
      const uniqueActivityTypes = [...new Set(
        calculatedData.rawData.map(row => row.Subtype).filter(Boolean)
      )];
      setActivityTypes(uniqueActivityTypes);

      // Extract unique years
      const uniqueYears = [...new Set(
        calculatedData.rawData.map(row => row.Year).filter(Boolean)
      )].sort();
      setYears(uniqueYears);
    }
  }, [calculatedData]);

  const handleFacilityToggle = (facility) => {
    const newFacilities = filters.facilities.includes(facility)
      ? filters.facilities.filter(f => f !== facility)
      : [...filters.facilities, facility];
    
    onFilterChange({ ...filters, facilities: newFacilities });
  };

  const handleActivityTypeToggle = (activityType) => {
    const newTypes = filters.activityTypes.includes(activityType)
      ? filters.activityTypes.filter(t => t !== activityType)
      : [...filters.activityTypes, activityType];
    
    onFilterChange({ ...filters, activityTypes: newTypes });
  };

  const handleYearChange = (year) => {
    onFilterChange({ ...filters, year });
  };

  return (
    <div className="bg-white border-b border-gray-200">
      <div className="max-w-[1400px] mx-auto px-8 py-4">
        {/* Toggle button */}
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-2 text-sm font-medium text-[#365D60] hover:text-[#5AB4AC]"
        >
          Filters
          <ChevronDown className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
        </button>

        {/* Filter content */}
        {isExpanded && (
          <div className="mt-4 flex flex-wrap gap-8">
            {/* Facilities Filter */}
            <div className="flex-1 min-w-[200px]">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Facilities</h3>
              <div className="flex flex-wrap gap-2">
                {facilities.map((facility) => (
                  <button
                    key={facility}
                    onClick={() => handleFacilityToggle(facility)}
                    className={`px-3 py-1.5 rounded-full text-sm transition-colors ${
                      filters.facilities.includes(facility)
                        ? 'bg-[#5AB4AC] text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {facility}
                  </button>
                ))}
                {facilities.length === 0 && (
                  <span className="text-sm text-gray-400">No facilities available</span>
                )}
              </div>
            </div>

            {/* Activity Type Filter */}
            <div className="flex-1 min-w-[200px]">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Activity Type</h3>
              <div className="flex flex-wrap gap-2">
                {activityTypes.map((type) => (
                  <button
                    key={type}
                    onClick={() => handleActivityTypeToggle(type)}
                    className={`px-3 py-1.5 rounded-full text-sm transition-colors ${
                      filters.activityTypes.includes(type)
                        ? 'bg-[#5AB4AC] text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {type}
                  </button>
                ))}
                {activityTypes.length === 0 && (
                  <span className="text-sm text-gray-400">No activity types available</span>
                )}
              </div>
            </div>

            {/* Year Filter */}
            <div className="flex-1 min-w-[200px]">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Year</h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleYearChange('all')}
                  className={`px-4 py-1.5 rounded-full text-sm transition-colors ${
                    filters.year === 'all'
                      ? 'bg-[#5AB4AC] text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  All
                </button>
                <select
                  value={filters.year === 'all' ? '' : filters.year}
                  onChange={(e) => handleYearChange(e.target.value || 'all')}
                  className="px-4 py-1.5 rounded-full text-sm bg-gray-100 hover:bg-gray-200 border-none outline-none cursor-pointer"
                >
                  <option value="">Select Year</option>
                  {years.map((year) => (
                    <option key={year} value={year}>
                      {year}
                    </option>
                  ))}
                </select>
                {years.length === 0 && filters.year === 'all' && (
                  <span className="text-sm text-gray-400">No years available</span>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FilterBar;