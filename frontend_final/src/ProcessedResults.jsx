import React, { useState } from 'react';

function SearchIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <circle cx="11" cy="11" r="8" strokeWidth="2"/>
      <path d="m21 21-4.35-4.35" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}

export default function ProcessedResults({
  onNext,
  onBack,
  sessionId,
  calculationResult,
  onGoToStep,
  onLogoClick,
}) {
  const [activeTab, setActiveTab] = useState('processed');
  const [selectedFile, setSelectedFile] = useState('efids');
  const [hoveredCell, setHoveredCell] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  // Mock fallback data — displayed in the row table (no backend endpoint for row-level records)
  const processedData = [
    { eq_equip_no: '1011', year: '2012', eqtyp_equip_type: 'FUEL CODE', in_service_date: '08/29/2012', dept_dept_code: 'SU099', loc_station_loc: '2012 FLTC SPU OFFROAD FUEL - SU099', fuel_type: 'UNL', mtco2: '0.0439', mtch4: '0.0000009', mtn2o: '0.0000010', mtco2e: '0.05', ftk_date: '06/26/2019' },
    { eq_equip_no: '1011', year: '2012', eqtyp_equip_type: 'FUEL CODE', in_service_date: '08/29/2012', dept_dept_code: 'SU099', loc_station_loc: '2012 FLTC SPU OFFROAD FUEL - SU099', fuel_type: 'UNL', mtco2: '0.0807', mtch4: '0.0000016', mtn2o: '0.0000018', mtco2e: '0.08', ftk_date: '10/15/2019' },
    { eq_equip_no: '1011', year: '2012', eqtyp_equip_type: 'FUEL CODE', in_service_date: '08/29/2012', dept_dept_code: 'SU099', loc_station_loc: '2012 FLTC SPU OFFROAD FUEL - SU099', fuel_type: 'UNL', mtco2: '0.0867', mtch4: '0.0000018', mtn2o: '0.0000019', mtco2e: '0.0881', ftk_date: '12/16/2020' },
    { eq_equip_no: '1011', year: '2012', eqtyp_equip_type: 'FUEL CODE', in_service_date: '08/29/2012', dept_dept_code: 'SU099', loc_station_loc: '2012 FLTC SPU OFFROAD FUEL - SU099', fuel_type: 'UNL', mtco2: '0.0878', mtch4: '0.0000018', mtn2o: '0.0000019', mtco2e: '0.09', ftk_date: '06/29/2021' },
    { eq_equip_no: '1011', year: '2012', eqtyp_equip_type: 'FUEL CODE', in_service_date: '08/29/2012', dept_dept_code: 'SU099', loc_station_loc: '2012 FLTC SPU OFFROAD FUEL - SU099', fuel_type: 'UNL', mtco2: '0.0878', mtch4: '0.0000018', mtn2o: '0.0000019', mtco2e: '0.09', ftk_date: '12/08/2021' },
    { eq_equip_no: '1021', year: '2017', eqtyp_equip_type: 'FUEL CODE', in_service_date: '08/15/2017', dept_dept_code: 'SU062', loc_station_loc: '2017 FLTC SPU OFFROAD FUEL - SU062', fuel_type: 'DEF', mtco2: '0.0000', mtch4: '0.0000000', mtn2o: '0.0000000', mtco2e: '0.00', ftk_date: '04/19/2019' },
    { eq_equip_no: '1021', year: '2017', eqtyp_equip_type: 'FUEL CODE', in_service_date: '08/15/2017', dept_dept_code: 'SU062', loc_station_loc: '2017 FLTC SPU OFFROAD FUEL - SU062', fuel_type: 'DEF', mtco2: '0.0000', mtch4: '0.0000000', mtn2o: '0.0000000', mtco2e: '0.00', ftk_date: '12/18/2019' },
    { eq_equip_no: '1021', year: '2017', eqtyp_equip_type: 'FUEL CODE', in_service_date: '08/15/2017', dept_dept_code: 'SU062', loc_station_loc: '2017 FLTC SPU', fuel_type: 'UNL', mtco2: '0.0433', mtch4: '0.0000009', mtn2o: '0.0000009', mtco2e: '0.0442', ftk_date: '03/02/2020' },
  ];

  // ── Derive summary values from real backend result ──────────────────────────
  const totalOutputRows = calculationResult?.results
    ? Object.values(calculationResult.results).reduce((s, r) => s + (r.outputRows || 0), 0)
    : processedData.length;

  const totalSources = calculationResult?.results
    ? Object.keys(calculationResult.results).length
    : [...new Set(processedData.map(r => r.eq_equip_no))].length;

  const totalMtCO2e = calculationResult?.aggregated?.totalMtCO2e != null
    ? calculationResult.aggregated.totalMtCO2e.toFixed(4)
    : processedData.reduce((s, r) => s + parseFloat(r.mtco2e), 0).toFixed(4);
  // ───────────────────────────────────────────────────────────────────────────

  const efidData = [
    { ef_id: 'Fuel_DSL_20XX_CO2_gal', ghg_mt: '0.0102100000', unit: 'CO2 MT/gal', ghg: 'CO2', gwp: '1', description: 'Fleet Combustion', scope: 'Scope 1' },
    { ef_id: 'Fuel_DSL_20XX_CH4_gal', ghg_mt: '0.0000002083', unit: 'CH4 MT/gal', ghg: 'CH4', gwp: '28', description: 'Fleet Combustion', scope: 'Scope 1' },
    { ef_id: 'Fuel_DSL_20XX_N2O_gal', ghg_mt: '0.0000002236', unit: 'N2O MT/gal', ghg: 'N2O', gwp: '265', description: 'Fleet Combustion', scope: 'Scope 1' },
    { ef_id: 'Fuel_BIO_20XX_CO2_gal', ghg_mt: '0.0094500000', unit: 'CO2 MT/gal', ghg: 'CO2', gwp: '1', description: 'Fleet Combustion', scope: 'Scope 1' },
    { ef_id: 'Fuel_BIO_20XX_CH4_gal', ghg_mt: '0.0000001928', unit: 'CH4 MT/gal', ghg: 'CH4', gwp: '28', description: 'Fleet Combustion', scope: 'Scope 1' },
    { ef_id: 'Fuel_BIO_20XX_N2O_gal', ghg_mt: '0.0000002070', unit: 'N2O MT/gal', ghg: 'N2O', gwp: '265', description: 'Fleet Combustion', scope: 'Scope 1' },
  ];

  const gwpData = [
    { ghg_id: 'CH4', ghg_name: 'CH4', ar5_gwp: '28', ghg_longname: 'Methane', ar2_sar: '21', ar3_tar: '23', ar4: '25', source: 'TCR' },
    { ghg_id: 'CO2', ghg_name: 'CO2', ar5_gwp: '1', ghg_longname: 'Carbon dioxide', ar2_sar: '1', ar3_tar: '1', ar4: '1', source: 'TCR' },
    { ghg_id: 'CO2e', ghg_name: 'CO2e', ar5_gwp: '1', ghg_longname: 'Carbon dioxide equivalent', ar2_sar: '1', ar3_tar: '1', ar4: '1', source: 'TCR' },
    { ghg_id: 'N2O', ghg_name: 'N2O', ar5_gwp: '265', ghg_longname: 'Nitrous oxide', ar2_sar: '310', ar3_tar: '296', ar4: '298', source: 'TCR' },
    { ghg_id: 'NF3', ghg_name: 'NF3', ar5_gwp: '16100', ghg_longname: 'Nitrogen trifluoride', ar2_sar: 'n/a', ar3_tar: '10800', ar4: '17200', source: 'TCR' },
    { ghg_id: 'R-134A', ghg_name: 'HFC-134A', ar5_gwp: '1300', ghg_longname: '1,1,1,2-tetrafluoroethane', ar2_sar: '1300', ar3_tar: '1300', ar4: '1430', source: 'TCR' },
    { ghg_id: 'R-22', ghg_name: 'HCFC-22', ar5_gwp: '1760', ghg_longname: 'Chlorodifluoromethane (Freon)', ar2_sar: '1500', ar3_tar: '1700', ar4: '1810', source: 'GHGP/CARB' },
  ];

  // Tooltip info for processed data rows
  const getRowInsight = (row) => {
    const co2e = parseFloat(row.mtco2e);
    if (co2e === 0) return { level: 'Zero', color: '#50B461', msg: 'No CO2e emissions recorded for this entry (DEF fuel type)' };
    if (co2e < 0.05) return { level: 'Very Low', color: '#29ABB5', msg: `Low emission: ${row.mtco2e} mtCO2e from ${row.fuel_type} fuel` };
    if (co2e < 0.09) return { level: 'Low', color: '#F0B429', msg: `Moderate: ${row.mtco2e} mtCO2e · Equipment #${row.eq_equip_no} · ${row.fuel_type}` };
    return { level: 'Moderate', color: '#E67E22', msg: `${row.mtco2e} mtCO2e from equipment #${row.eq_equip_no} (${row.fuel_type})` };
  };

  const handleCellHover = (e, rowIdx, row) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setTooltipPos({ x: rect.left + rect.width / 2, y: rect.top - 8 });
    setHoveredCell({ rowIdx, row });
  };

  const thStyle = "px-[16px] py-[12px] text-left font-['Noto_Sans',sans-serif] font-medium text-[14px] text-[#13383B] whitespace-nowrap";
  const tdStyle = "px-[16px] py-[12px] font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B]";
  const handleLogoClick = () => {
    if (onLogoClick) onLogoClick();
  };
  const handleStepClick = (step, fallback) => {
    if (onGoToStep) {
      onGoToStep(step);
      return;
    }
    if (fallback) fallback();
  };

  return (
    <div className="bg-white relative w-full min-h-screen overflow-auto">
      {/* Toolbar */}
      <div className="sticky top-0 left-0 w-full h-[62px] bg-[#365D60] flex items-center justify-between px-[64px] z-20">
        <button
          type="button"
          onClick={handleLogoClick}
          className="flex items-center gap-[12px] bg-transparent border-0 p-0 cursor-pointer"
        >
          <img src="/logo.png" alt="Logo" className="h-[40px] w-auto" />
          <p className="font-['Noto_Sans',sans-serif] font-bold text-[24px] text-white">EcoMetrics</p>
        </button>
        <div className="bg-white rounded-[10px] w-[331px] h-[40px] flex items-center px-[12px]">
          <SearchIcon className="w-[20px] h-[20px] text-[#7E7E7E]" />
        </div>
      </div>

      {/* Progress Steps */}
      <div className="sticky top-[62px] bg-white left-[64px] right-[64px] py-[19px] px-[64px] flex items-center z-10 border-b border-[#EEEEEE]">
        <div className="flex items-center gap-[12px] cursor-pointer" onClick={() => handleStepClick(1)}>
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-white">✓</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">Upload Your Data</p>
        </div>
        <div className="flex-1 h-[2px] bg-[#50B461] mx-[16px]"></div>
        <div className="flex items-center gap-[12px] cursor-pointer" onClick={() => handleStepClick(3)}>
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-white">✓</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">Data Cleaning</p>
        </div>
        <div className="flex-1 h-[2px] bg-[#50B461] mx-[16px]"></div>
        <div className="flex items-center gap-[12px] cursor-pointer" onClick={() => handleStepClick(5, onBack)}>
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-white">✓</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">Data Processing</p>
        </div>
        <div className="flex-1 h-[2px] bg-[#29ABB5] mx-[16px]"></div>
        <div className="flex items-center gap-[12px] cursor-pointer" onClick={() => handleStepClick(6, onNext)}>
          <div className="w-[32px] h-[32px] rounded-full bg-white border-2 border-[#ACC2C5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-[#7E7E7E]">4</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E] whitespace-nowrap">Data Visualization</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="px-[64px] py-[32px] pb-[100px]">
        <h1 className="font-['Noto_Sans',sans-serif] font-bold text-[26px] text-[#13383B] mb-[24px]">
          Processed Results
        </h1>

        {/* Summary Cards */}
        <div className="grid grid-cols-3 gap-[16px] mb-[24px]">
          <div className="bg-gradient-to-r from-[#E8F4F3] to-[#D4EBE9] rounded-[10px] p-[16px]">
            <p className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E] mb-[4px]">Total Records</p>
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[24px] text-[#365D60]">{totalOutputRows.toLocaleString()}</p>
          </div>
          <div className="bg-gradient-to-r from-[#FFF8E1] to-[#FFF3CD] rounded-[10px] p-[16px]">
            <p className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E] mb-[4px]">Sources Processed</p>
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[24px] text-[#E6A817]">
              {totalSources}
            </p>
          </div>
          <div className="bg-gradient-to-r from-[#F3E8F9] to-[#EBD6F5] rounded-[10px] p-[16px]">
            <p className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E] mb-[4px]">Total mtCO2e</p>
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[24px] text-[#9B59B6]">
              {totalMtCO2e}
            </p>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-[12px] mb-[24px]">
          <button onClick={() => setActiveTab('processed')}
            className={`px-[24px] py-[12px] rounded-[8px] font-['Noto_Sans',sans-serif] font-medium text-[14px] transition ${
              activeTab === 'processed' ? 'bg-[#29ABB5] text-white' : 'bg-[#F5F5F5] text-[#7E7E7E] hover:bg-[#EEEEEE]'
            }`}>Processed Results</button>
          <button onClick={() => setActiveTab('raw')}
            className={`px-[24px] py-[12px] rounded-[8px] font-['Noto_Sans',sans-serif] font-medium text-[14px] transition ${
              activeTab === 'raw' ? 'bg-[#29ABB5] text-white' : 'bg-[#29ABB5]/20 text-[#365D60] hover:bg-[#29ABB5]/30'
            }`}>Raw Datasets</button>
        </div>

        {/* Processed Results */}
        {activeTab === 'processed' && (
          <div className="bg-white rounded-[10px] border border-[#EEEEEE] overflow-hidden mb-[24px]">
            <div className="overflow-x-auto max-h-[450px]">
              <table className="w-full">
                <thead className="bg-[#D9D9D9] sticky top-0 z-[5]">
                  <tr>
                    <th className={thStyle}>EQ_EQUIP_NO</th>
                    <th className={thStyle}>YEAR</th>
                    <th className={thStyle}>EQTYP_EQUIP_TYPE</th>
                    <th className={thStyle}>IN_SERVICE_DATE</th>
                    <th className={thStyle}>DEPT_DEPT_CODE</th>
                    <th className={thStyle}>LOC_STATION_LOC</th>
                    <th className={thStyle}>FUEL_TYPE</th>
                    <th className={thStyle}>mtCO2</th>
                    <th className={thStyle}>mtCH4</th>
                    <th className={thStyle}>mtN2O</th>
                    <th className={thStyle}>mtCO2e</th>
                    <th className={thStyle}>FTK_DATE</th>
                  </tr>
                </thead>
                <tbody>
                  {processedData.map((row, idx) => {
                    const insight = getRowInsight(row);
                    const isHovered = hoveredCell?.rowIdx === idx;
                    return (
                      <tr key={idx} 
                        className={`border-t border-[#EEEEEE] transition-all cursor-pointer ${isHovered ? 'bg-[#E8F4F3]' : 'hover:bg-[#F5F5F5]'}`}
                        onMouseEnter={(e) => handleCellHover(e, idx, row)}
                        onMouseLeave={() => setHoveredCell(null)}>
                        <td className={tdStyle}>{row.eq_equip_no}</td>
                        <td className={tdStyle}>{row.year}</td>
                        <td className={tdStyle}>{row.eqtyp_equip_type}</td>
                        <td className={tdStyle}>{row.in_service_date}</td>
                        <td className={tdStyle}>{row.dept_dept_code}</td>
                        <td className={`${tdStyle} max-w-[200px] truncate`}>{row.loc_station_loc}</td>
                        <td className={tdStyle}>
                          <span className={`px-[8px] py-[2px] rounded-[4px] text-[12px] font-medium ${
                            row.fuel_type === 'UNL' ? 'bg-[#29ABB5]/15 text-[#29ABB5]' :
                            row.fuel_type === 'DEF' ? 'bg-[#F0B429]/15 text-[#D49A17]' :
                            'bg-[#E3E3E3] text-[#575757]'
                          }`}>{row.fuel_type}</span>
                        </td>
                        <td className={tdStyle}>{row.mtco2}</td>
                        <td className={tdStyle}>{row.mtch4}</td>
                        <td className={tdStyle}>{row.mtn2o}</td>
                        <td className={tdStyle}>
                          <span className="font-medium" style={{ color: insight.color }}>{row.mtco2e}</span>
                        </td>
                        <td className={tdStyle}>{row.ftk_date}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Floating Tooltip */}
        {hoveredCell && activeTab === 'processed' && (
          <div className="fixed z-50 pointer-events-none" 
            style={{ left: Math.min(tooltipPos.x, window.innerWidth - 340), top: tooltipPos.y, transform: 'translate(-50%, -100%)' }}>
            <div className="bg-[#365D60] text-white px-[16px] py-[12px] rounded-[8px] shadow-xl max-w-[320px]"
              style={{ animation: 'tooltipIn 0.15s ease-out' }}>
              <div className="flex items-center gap-[8px] mb-[6px]">
                <div className="w-[8px] h-[8px] rounded-full" style={{ backgroundColor: getRowInsight(hoveredCell.row).color }}/>
                <span className="text-[12px] font-bold">{getRowInsight(hoveredCell.row).level} Emission</span>
              </div>
              <p className="text-[11px] opacity-90 leading-[1.4]">{getRowInsight(hoveredCell.row).msg}</p>
              <div className="text-[10px] opacity-60 mt-[4px]">
                Equip #{hoveredCell.row.eq_equip_no} · {hoveredCell.row.ftk_date}
              </div>
              <div className="absolute bottom-[-6px] left-1/2 -translate-x-1/2 w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[6px] border-t-[#365D60]"/>
            </div>
          </div>
        )}

        {/* Raw Datasets */}
        {activeTab === 'raw' && (
          <>
            <div className="flex gap-[12px] mb-[16px]">
              <button onClick={() => setSelectedFile('efids')}
                className={`px-[16px] py-[8px] rounded-[4px] font-['Noto_Sans',sans-serif] text-[14px] transition ${
                  selectedFile === 'efids' ? 'bg-[#365D60] text-white' : 'bg-[#E3E3E3] text-[#575757] hover:bg-[#D8D8D8]'
                }`}>EFIDs.xlsx</button>
              <button onClick={() => setSelectedFile('gwp')}
                className={`px-[16px] py-[8px] rounded-[4px] font-['Noto_Sans',sans-serif] text-[14px] transition ${
                  selectedFile === 'gwp' ? 'bg-[#365D60] text-white' : 'bg-[#E3E3E3] text-[#575757] hover:bg-[#D8D8D8]'
                }`}>GWP.xlsx</button>
              <button className="ml-auto px-[16px] py-[8px] rounded-[4px] font-['Noto_Sans',sans-serif] text-[14px] bg-[#E3E3E3] text-[#575757] hover:bg-[#D8D8D8] transition">
                Edit
              </button>
            </div>
            <div className="bg-white rounded-[10px] border border-[#EEEEEE] overflow-hidden mb-[24px]">
              <div className="overflow-x-auto max-h-[400px]">
                {selectedFile === 'efids' ? (
                  <table className="w-full">
                    <thead className="bg-[#D9D9D9] sticky top-0">
                      <tr>
                        {['EF_ID','GHG_MTperUnit','Unit','GHG','GWP','Description','Scope'].map(h => (
                          <th key={h} className={thStyle}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {efidData.map((row, idx) => (
                        <tr key={idx} className="border-t border-[#EEEEEE] hover:bg-[#E8F4F3] transition">
                          <td className={tdStyle}>{row.ef_id}</td>
                          <td className={tdStyle}>{row.ghg_mt}</td>
                          <td className={tdStyle}>{row.unit}</td>
                          <td className={tdStyle}>{row.ghg}</td>
                          <td className={tdStyle}>{row.gwp}</td>
                          <td className={tdStyle}>{row.description}</td>
                          <td className={tdStyle}>{row.scope}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <table className="w-full">
                    <thead className="bg-[#D9D9D9] sticky top-0">
                      <tr>
                        {['GHG_ID','GHG_Name','AR5_GWP','GHG_LongName','AR2_SAR','AR3_TAR','AR4','Source'].map(h => (
                          <th key={h} className={thStyle}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {gwpData.map((row, idx) => (
                        <tr key={idx} className="border-t border-[#EEEEEE] hover:bg-[#E8F4F3] transition">
                          <td className={tdStyle}>{row.ghg_id}</td>
                          <td className={tdStyle}>{row.ghg_name}</td>
                          <td className={tdStyle}>{row.ar5_gwp}</td>
                          <td className={tdStyle}>{row.ghg_longname}</td>
                          <td className={tdStyle}>{row.ar2_sar}</td>
                          <td className={tdStyle}>{row.ar3_tar}</td>
                          <td className={tdStyle}>{row.ar4}</td>
                          <td className={tdStyle}>{row.source}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Dashboard Button - Fixed at bottom */}
      <div className="fixed bottom-[32px] right-[64px] z-30">
        <button onClick={onNext}
          className="flex items-center gap-[12px] px-[32px] py-[14px] bg-[#29ABB5] rounded-[24.5px] shadow-[0px_4px_8px_0px_rgba(0,0,0,0.15)] font-['Noto_Sans',sans-serif] font-bold text-[18px] text-white hover:bg-[#238d96] transition"
          style={{ animation: 'slideUp 0.5s ease-out' }}>
          Dashboard
          <svg className="w-[20px] h-[20px]" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M5 12h14M12 5l7 7-7 7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>

      <style>{`
        @keyframes tooltipIn { from { opacity: 0; transform: translate(-50%, -90%); } to { opacity: 1; transform: translate(-50%, -100%); } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}
