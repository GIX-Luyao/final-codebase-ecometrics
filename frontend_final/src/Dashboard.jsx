import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { getAnalyticsSummary, getChartData } from './services/api';

// ═══════════════════════════════════════════════
// Dashboard Data (from dashboard_data.json)
// ═══════════════════════════════════════════════
const DASHBOARD_DATA = {
  byYear: { "2019": 6881.01, "2020": 6454.00, "2021": 6486.42, "2022": 12681.80 },
  bySourceYear: {
    Fleet: { "2019": 5036.87, "2020": 4489.49, "2021": 4898.36, "2022": 4631.48 },
    PSE: { "2019": 0, "2020": 0, "2021": 0, "2022": 6348.65 },
    SCL: { "2019": 0, "2020": 0, "2021": 0, "2022": 250.64 },
    Landfill: { "2019": 1844.14, "2020": 1964.51, "2021": 1588.06, "2022": 1451.02 }
  },
  byCommodityYear: {
    "2019": { B20: 3400.56, UNL: 787.93, R9B: 444.62, DSL: 371.69 },
    "2020": { B20: 2378.08, R9B: 1116.91, UNL: 739.04, R8B: 142.24 },
    "2021": { B20: 3925.31, UNL: 937.48, PRO: 16.97, DSL: 16.55 },
    "2022": { B20: 3745.40, UNL: 860.63, DSL: 17.70, PRO: 7.28 }
  },
  totals: { Fleet: 19056.20, PSE: 6348.65, SCL: 250.64, Landfill: 6847.73, Total: 32503.23 }
};

const ACTIVITY_MAPPING = {
  'Fossil Fuel': ['Fleet'], 'Gas': ['Fleet'], 'Electricity': ['PSE', 'SCL'],
  'Landfill': ['Landfill'], 'Fleet': ['Fleet']
};

// ═══════════════════════════════════════════════
// Icons
// ═══════════════════════════════════════════════
const Icon = ({ d, className, viewBox = "0 0 24 24" }) => (
  <svg className={className} viewBox={viewBox} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    {typeof d === 'string' ? <path d={d}/> : d}
  </svg>
);
const SearchIcon = ({ className }) => <Icon className={className} d={<><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></>}/>;
const FilterIcon = ({ className }) => <Icon className={className} d={<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" strokeLinejoin="round"/>}/>;
const XIcon = ({ className }) => <Icon className={className} d={<><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></>}/>;
const ChevronLeftIcon = ({ className }) => <Icon className={className} d="M15 18l-6-6 6-6"/>;
const ChevronRightIcon = ({ className }) => <Icon className={className} d="M9 18l6-6-6-6"/>;
const PlusIcon = ({ className }) => <Icon className={className} d={<><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></>}/>;
const TrashIcon = ({ className }) => <Icon className={className} d={<><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></>}/>;
const BarChartIcon = ({ className }) => <Icon className={className} d={<><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></>}/>;
const LineChartIcon = ({ className }) => <Icon className={className} d="M22 12 18 12 15 21 9 3 6 12 2 12"/>;
const PieChartIcon = ({ className }) => <Icon className={className} d={<><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></>}/>;
const StackedBarIcon = ({ className }) => <Icon className={className} d={<><rect x="4" y="14" width="4" height="6" rx="0.5"/><rect x="4" y="8" width="4" height="5" rx="0.5"/><rect x="10" y="10" width="4" height="10" rx="0.5"/><rect x="10" y="4" width="4" height="5" rx="0.5"/><rect x="16" y="12" width="4" height="8" rx="0.5"/><rect x="16" y="6" width="4" height="5" rx="0.5"/></>}/>;
const AreaChartIcon = ({ className }) => <Icon className={className} d={<><path d="M3 20 7 14 11 16 16 8 21 12 21 20Z" fill="currentColor" opacity="0.2"/><polyline points="3 20 7 14 11 16 16 8 21 12"/></>}/>;
const GridIcon = ({ className }) => <Icon className={className} d={<><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>}/>;
const TrendUpIcon = ({ className }) => <Icon className={className} d={<><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></>}/>;
const MoveIcon = ({ className }) => <Icon className={className} d={<><polyline points="5 9 2 12 5 15"/><polyline points="9 5 12 2 15 5"/><polyline points="15 19 12 22 9 19"/><polyline points="19 9 22 12 19 15"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="12" y1="2" x2="12" y2="22"/></>}/>;
const MaximizeIcon = ({ className }) => <Icon className={className} d={<><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></>}/>;
const LayoutIcon = ({ className }) => <Icon className={className} d={<><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></>}/>;
const LockIcon = ({ className }) => <Icon className={className} d={<><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></>}/>;
const UnlockIcon = ({ className }) => <Icon className={className} d={<><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/></>}/>;

// ═══════════════════════════════════════════════
// Color Palettes
// ═══════════════════════════════════════════════
const COLOR_PALETTES = {
  teal: ['#2A9D8F', '#40B4A6', '#6ECCC1', '#A8E0DA', '#D4F0ED'],
  gold: ['#E6A817', '#F0B429', '#F5C94D', '#F9DD7A', '#FDF2A8'],
  purple: ['#7B2D8E', '#9B59B6', '#AF7AC5', '#C39BD3', '#D7BDE2'],
  rose: ['#C2185B', '#E91E63', '#F06292', '#F48FB1', '#F8BBD0'],
};

// ═══════════════════════════════════════════════
// SVG Chart Renderers (unchanged)
// ═══════════════════════════════════════════════
function PieChartSVG({ labels, values, colors, size = 180, donut = false, onHover }) {
  if (!labels?.length) return <p className="text-[#7E7E7E] text-sm">No data</p>;
  const total = values.reduce((s, v) => s + v, 0);
  let angle = -90;
  const r = size / 2 - 10;
  const ir = donut ? r * 0.55 : 0;
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="w-full" style={{ maxHeight: size }}>
      <g transform={`translate(${size/2}, ${size/2})`}>
        {values.map((value, idx) => {
          const pct = total > 0 ? value / total : 0;
          const sweep = pct * 360;
          const r1 = (angle * Math.PI) / 180, r2 = ((angle + sweep) * Math.PI) / 180;
          const x1o = r * Math.cos(r1), y1o = r * Math.sin(r1);
          const x2o = r * Math.cos(r2), y2o = r * Math.sin(r2);
          const x1i = ir * Math.cos(r1), y1i = ir * Math.sin(r1);
          const x2i = ir * Math.cos(r2), y2i = ir * Math.sin(r2);
          const large = sweep > 180 ? 1 : 0;
          const path = donut
            ? `M ${x1o} ${y1o} A ${r} ${r} 0 ${large} 1 ${x2o} ${y2o} L ${x2i} ${y2i} A ${ir} ${ir} 0 ${large} 0 ${x1i} ${y1i} Z`
            : `M 0 0 L ${x1o} ${y1o} A ${r} ${r} 0 ${large} 1 ${x2o} ${y2o} Z`;
          angle += sweep;
          return <path key={idx} d={path} fill={colors[idx % colors.length]} stroke="white" strokeWidth="2"
            className="cursor-pointer transition-opacity hover:opacity-80"
            onMouseEnter={() => onHover?.({ label: labels[idx], value, pct: pct * 100 })} onMouseLeave={() => onHover?.(null)}/>;
        })}
        {donut && <text textAnchor="middle" dy="6" className="text-[14px] font-bold fill-[#365D60]">{total.toFixed(0)}</text>}
      </g>
    </svg>
  );
}

function BarChartSVG({ labels, values, colors, stacked, stackData, onHover }) {
  if (!labels?.length) return <p className="text-[#7E7E7E] text-sm">No data</p>;
  const W = 400, H = 200, PL = 50, PR = 20, PT = 10, PB = 40;
  if (stacked && stackData) {
    const maxVal = Math.max(...labels.map((_, i) => Object.values(stackData).reduce((s, src) => s + (src[labels[i]] || 0), 0)), 1);
    const sources = Object.keys(stackData);
    const barW = (W - PL - PR) / labels.length - 8;
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: H }}>
        <line x1={PL} y1={PT} x2={PL} y2={H-PB} stroke="#D9D9D9" strokeWidth="1"/>
        <line x1={PL} y1={H-PB} x2={W-PR} y2={H-PB} stroke="#D9D9D9" strokeWidth="1"/>
        {labels.map((label, i) => {
          const x = PL + 4 + i * ((W - PL - PR) / labels.length); let cumY = 0;
          return (<g key={i}>
            {sources.map((src, si) => {
              const val = stackData[src]?.[label] || 0;
              const barH = (val / maxVal) * (H - PT - PB); const y = H - PB - cumY - barH; cumY += barH;
              return <rect key={si} x={x} y={y} width={barW} height={Math.max(barH, 0)} fill={colors[si % colors.length]} rx="1"
                className="cursor-pointer transition-opacity hover:opacity-80"
                onMouseEnter={() => onHover?.({ label: `${src} (${label})`, value: val })} onMouseLeave={() => onHover?.(null)}/>;
            })}
            <text x={x + barW / 2} y={H-PB+14} textAnchor="middle" className="text-[9px] fill-[#575757]">{label}</text>
          </g>);
        })}
        <text x="6" y={H/2} className="text-[8px] fill-[#7E7E7E]" transform={`rotate(-90,6,${H/2})`}>mtCO2e</text>
      </svg>
    );
  }
  const maxVal = Math.max(...values, 1);
  const barW = (W - PL - PR) / values.length - 8;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: H }}>
      <line x1={PL} y1={PT} x2={PL} y2={H-PB} stroke="#D9D9D9" strokeWidth="1"/>
      <line x1={PL} y1={H-PB} x2={W-PR} y2={H-PB} stroke="#D9D9D9" strokeWidth="1"/>
      {values.map((v, i) => {
        const barH = (v / maxVal) * (H - PT - PB); const x = PL + 4 + i * ((W - PL - PR) / values.length);
        return (<g key={i}>
          <rect x={x} y={H-PB-barH} width={barW} height={barH} fill={colors[i % colors.length]} rx="2"
            className="cursor-pointer transition-opacity hover:opacity-80"
            onMouseEnter={() => onHover?.({ label: labels[i], value: v })} onMouseLeave={() => onHover?.(null)}/>
          <text x={x + barW/2} y={H-PB+14} textAnchor="middle" className="text-[9px] fill-[#575757]">{labels[i]}</text>
        </g>);
      })}
      <text x="6" y={H/2} className="text-[8px] fill-[#7E7E7E]" transform={`rotate(-90,6,${H/2})`}>mtCO2e</text>
    </svg>
  );
}

function LineChartSVG({ labels, values, colors, area = false, onHover }) {
  if (!labels?.length) return <p className="text-[#7E7E7E] text-sm">No data</p>;
  const W = 400, H = 200, PL = 50, PR = 20, PT = 20, PB = 40;
  const maxVal = Math.max(...values, 1);
  const cx = i => PL + (i / Math.max(values.length - 1, 1)) * (W - PL - PR);
  const cy = v => PT + (H - PT - PB) * (1 - v / maxVal);
  const pts = values.map((v, i) => `${cx(i)},${cy(v)}`).join(' ');
  const color = colors[0];
  const areaPath = area ? `M ${cx(0)},${H-PB} ${values.map((v, i) => `L ${cx(i)},${cy(v)}`).join(' ')} L ${cx(values.length-1)},${H-PB} Z` : '';
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: H }}>
      <line x1={PL} y1={PT} x2={PL} y2={H-PB} stroke="#D9D9D9" strokeWidth="1"/>
      <line x1={PL} y1={H-PB} x2={W-PR} y2={H-PB} stroke="#D9D9D9" strokeWidth="1"/>
      {area && <path d={areaPath} fill={color} opacity="0.15"/>}
      <polyline fill="none" stroke={color} strokeWidth="2.5" points={pts} strokeLinejoin="round"/>
      {values.map((v, i) => (
        <g key={i}>
          <circle cx={cx(i)} cy={cy(v)} r="5" fill={color} stroke="white" strokeWidth="2" className="cursor-pointer"
            onMouseEnter={() => onHover?.({ label: labels[i], value: v })} onMouseLeave={() => onHover?.(null)}/>
          <text x={cx(i)} y={H-PB+14} textAnchor="middle" className="text-[9px] fill-[#575757]">{labels[i]}</text>
        </g>
      ))}
      <text x="6" y={H/2} className="text-[8px] fill-[#7E7E7E]" transform={`rotate(-90,6,${H/2})`}>mtCO2e</text>
    </svg>
  );
}

function HBarChartSVG({ labels, values, colors, onHover }) {
  if (!labels?.length) return <p className="text-[#7E7E7E] text-sm">No data</p>;
  const W = 400, H = Math.max(labels.length * 32 + 20, 160);
  const maxVal = Math.max(...values, 1); const PL = 100, PR = 40, barH = 20;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: H }}>
      {labels.map((label, i) => {
        const barW = (values[i] / maxVal) * (W - PL - PR); const y = 10 + i * 32;
        return (<g key={i}>
          <text x={PL-8} y={y+barH/2+4} textAnchor="end" className="text-[9px] fill-[#575757]">{label}</text>
          <rect x={PL} y={y} width={Math.max(barW,2)} height={barH} fill={colors[i % colors.length]} rx="2"
            className="cursor-pointer transition-opacity hover:opacity-80"
            onMouseEnter={() => onHover?.({ label, value: values[i] })} onMouseLeave={() => onHover?.(null)}/>
          <text x={PL+barW+6} y={y+barH/2+4} className="text-[8px] fill-[#575757]">{values[i].toFixed(0)}</text>
        </g>);
      })}
    </svg>
  );
}

function ChartRenderer({ chart, onHover }) {
  const { type, data, colors } = chart;
  if (type === 'pie') return <PieChartSVG {...data} colors={colors} onHover={onHover}/>;
  if (type === 'donut') return <PieChartSVG {...data} colors={colors} donut onHover={onHover}/>;
  if (type === 'bar') return <BarChartSVG {...data} colors={colors} onHover={onHover}/>;
  if (type === 'stacked-bar') return <BarChartSVG {...data} colors={colors} stacked onHover={onHover}/>;
  if (type === 'line') return <LineChartSVG {...data} colors={colors} onHover={onHover}/>;
  if (type === 'area') return <LineChartSVG {...data} colors={colors} area onHover={onHover}/>;
  if (type === 'hbar') return <HBarChartSVG {...data} colors={colors} onHover={onHover}/>;
  return <p className="text-[#7E7E7E] text-sm">Unknown</p>;
}

const CHART_TYPES = [
  { id: 'bar', icon: BarChartIcon, label: 'Bar' },
  { id: 'stacked-bar', icon: StackedBarIcon, label: 'Stacked' },
  { id: 'line', icon: LineChartIcon, label: 'Line' },
  { id: 'area', icon: AreaChartIcon, label: 'Area' },
  { id: 'pie', icon: PieChartIcon, label: 'Pie' },
  { id: 'donut', icon: PieChartIcon, label: 'Donut' },
  { id: 'hbar', icon: BarChartIcon, label: 'H-Bar' },
];

// ═══════════════════════════════════════════════
// Overall Emissions Banner (CLICK-only modes)
// ═══════════════════════════════════════════════
function OverallEmissionsBanner({ data }) {
  const [displayMode, setDisplayMode] = useState(0);
  const modes = ['Facilities', 'Activity Type', 'Year'];
  const total = data.totals.Total, pseTotal = data.totals.PSE, sclTotal = data.totals.SCL;
  const fleetTotal = data.totals.Fleet, landfillTotal = data.totals.Landfill;
  const pctChange = ((data.byYear['2022'] - data.byYear['2021']) / data.byYear['2021'] * 100);

  const renderBreakdown = () => {
    if (displayMode === 0) {
      return (
        <div className="space-y-[10px]" key="fac" style={{ animation: 'fadeSlide 0.35s ease-out' }}>
          {[{ label: 'SCL', value: sclTotal, color: '#2A9D8F' }, { label: 'PSE', value: pseTotal, color: '#29ABB5' }].map(item => (
            <div key={item.label} className="flex items-center gap-[12px]">
              <span className="text-[13px] text-[#365D60] font-medium w-[40px]">{item.label}</span>
              <div className="flex-1 bg-white/50 rounded-full h-[18px] overflow-hidden">
                <div className="h-full rounded-full transition-all duration-700" style={{ width: `${Math.max((item.value / total) * 100, 1)}%`, backgroundColor: item.color }}/>
              </div>
              <span className="text-[13px] text-[#365D60] font-bold w-[130px] text-right">{item.value.toFixed(1)} MTCO2e</span>
            </div>
          ))}
        </div>
      );
    }
    if (displayMode === 1) {
      const items = [
        { label: 'Fleet', value: fleetTotal, color: '#2A9D8F' },
        { label: 'Electricity', value: pseTotal + sclTotal, color: '#7B2D8E' },
        { label: 'Landfill', value: landfillTotal, color: '#E6A817' },
      ];
      return (
        <div className="space-y-[8px]" key="act" style={{ animation: 'fadeSlide 0.35s ease-out' }}>
          {items.map(item => (
            <div key={item.label} className="flex items-center gap-[12px]">
              <span className="text-[12px] text-[#365D60] font-medium w-[70px]">{item.label}</span>
              <div className="flex-1 bg-white/50 rounded-full h-[14px] overflow-hidden">
                <div className="h-full rounded-full transition-all duration-700" style={{ width: `${(item.value / total) * 100}%`, backgroundColor: item.color }}/>
              </div>
              <span className="text-[12px] text-[#365D60] font-bold w-[100px] text-right">{item.value.toFixed(0)}</span>
            </div>
          ))}
        </div>
      );
    }
    return (
      <div className="space-y-[6px]" key="yr" style={{ animation: 'fadeSlide 0.35s ease-out' }}>
        {Object.entries(data.byYear).map(([year, val]) => (
          <div key={year} className="flex items-center gap-[12px]">
            <span className="text-[12px] text-[#365D60] font-medium w-[40px]">{year}</span>
            <div className="flex-1 bg-white/50 rounded-full h-[12px] overflow-hidden">
              <div className="h-full bg-[#C2185B] rounded-full transition-all duration-700" style={{ width: `${(val / Math.max(...Object.values(data.byYear))) * 100}%` }}/>
            </div>
            <span className="text-[11px] text-[#365D60] font-bold w-[80px] text-right">{val.toFixed(0)}</span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="bg-gradient-to-r from-[#E8F4F3] to-[#D4EBE9] rounded-[10px] p-[24px] mb-[20px]">
      <div className="grid grid-cols-[1fr_1fr] gap-[32px] items-center">
        <div>
          <div className="flex items-center gap-[12px] mb-[8px]">
            <p className="text-[14px] text-[#7E7E7E] font-medium">Overall Emissions</p>
            <span className={`flex items-center gap-[4px] text-[12px] px-[8px] py-[2px] rounded-[4px] ${pctChange > 0 ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'}`}>
              <TrendUpIcon className="w-[12px] h-[12px]"/>{Math.abs(pctChange).toFixed(1)}%
            </span>
          </div>
          <div className="flex items-baseline gap-[12px]">
            <span className="font-bold text-[44px] text-[#2C5F7C] leading-none">{total.toFixed(1)}</span>
            <span className="text-[18px] text-[#7E7E7E]">MTCO2e</span>
          </div>
        </div>
        <div>
          <div className="flex items-center gap-[6px] mb-[12px]">
            {modes.map((mode, i) => (
              <button key={mode} onClick={() => setDisplayMode(i)}
                className={`text-[11px] px-[12px] py-[4px] rounded-full transition-all duration-200 cursor-pointer ${
                  displayMode === i ? 'bg-[#365D60] text-white shadow-sm' : 'text-[#365D60] hover:bg-white/60 bg-white/30'
                }`}>{mode}</button>
            ))}
          </div>
          {renderBreakdown()}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════
// Chart Convert Overlay (unchanged)
// ═══════════════════════════════════════════════
function ChartConvertOverlay({ chart, onGenerate, onClose }) {
  const available = CHART_TYPES.filter(t => t.id !== chart.type);
  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center" onClick={onClose} style={{ animation: 'fadeIn 0.15s ease-out' }}>
      <div className="absolute inset-0 bg-black/50"/>
      <div className="relative bg-white rounded-[12px] p-[24px] shadow-2xl w-[420px]" onClick={e => e.stopPropagation()} style={{ animation: 'slideUp 0.25s ease-out' }}>
        <div className="flex items-center justify-between mb-[14px]">
          <h3 className="font-bold text-[16px] text-[#13383B]">Generate New Chart</h3>
          <button onClick={onClose} className="p-[4px] hover:bg-[#F5F5F5] rounded-[6px]"><XIcon className="w-[16px] h-[16px] text-[#575757]"/></button>
        </div>
        <p className="text-[12px] text-[#7E7E7E] mb-[14px]">From "<strong>{chart.title}</strong>" — pick a chart type:</p>
        <div className="grid grid-cols-4 gap-[8px]">
          {available.map(type => {
            const TypeIcon = type.icon;
            return (
              <button key={type.id} onClick={() => { onGenerate(chart, type.id); onClose(); }}
                className="flex flex-col items-center gap-[4px] p-[10px] rounded-[8px] border-2 border-[#EEEEEE] hover:border-[#29ABB5] hover:bg-[#E8F4F3] transition-all duration-200 cursor-pointer">
                <TypeIcon className="w-[20px] h-[20px] text-[#365D60]"/><span className="text-[10px] text-[#575757]">{type.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════
// Detail View with Sliding Navigation (unchanged)
// ═══════════════════════════════════════════════
function ChartDetailView({ chart, allCharts, onClose, onNavigate, onConvert }) {
  const [hovered, setHovered] = useState(null);
  const [showTypeSelector, setShowTypeSelector] = useState(false);
  const currentIdx = allCharts.findIndex(c => c.id === chart.id);
  const hasPrev = currentIdx > 0, hasNext = currentIdx < allCharts.length - 1;
  const insights = useMemo(() => {
    if (!chart.data?.values) return [];
    const vals = chart.data.values, labels = chart.data.labels || [];
    const maxIdx = vals.indexOf(Math.max(...vals)), minIdx = vals.indexOf(Math.min(...vals));
    const total = vals.reduce((s, v) => s + v, 0);
    return [
      `Highest: ${labels[maxIdx]} (${vals[maxIdx]?.toFixed(1)} mtCO2e, ${((vals[maxIdx] / total) * 100).toFixed(1)}%)`,
      `Lowest: ${labels[minIdx]} (${vals[minIdx]?.toFixed(1)} mtCO2e)`,
      `Total: ${total.toFixed(1)} mtCO2e across ${vals.length} categories`,
    ];
  }, [chart]);

  return (
    <div className="fixed inset-0 bg-black/70 z-[80] flex" style={{ animation: 'fadeIn 0.2s ease-out' }}>
      {hasPrev && <button onClick={() => onNavigate(allCharts[currentIdx - 1])} className="absolute left-[16px] top-1/2 -translate-y-1/2 w-[44px] h-[44px] bg-white/90 rounded-full flex items-center justify-center shadow-lg hover:bg-white transition z-[90]"><ChevronLeftIcon className="w-[24px] h-[24px] text-[#365D60]"/></button>}
      {hasNext && <button onClick={() => onNavigate(allCharts[currentIdx + 1])} className="absolute right-[16px] top-1/2 -translate-y-1/2 w-[44px] h-[44px] bg-white/90 rounded-full flex items-center justify-center shadow-lg hover:bg-white transition z-[90]"><ChevronRightIcon className="w-[24px] h-[24px] text-[#365D60]"/></button>}
      <div className="flex-1 flex items-center justify-center p-[60px]" onClick={onClose}>
        <div className="bg-white rounded-[12px] w-full max-w-[900px] max-h-[90vh] overflow-auto shadow-2xl" onClick={e => e.stopPropagation()} style={{ animation: 'slideUp 0.3s ease-out' }}>
          <div className="flex items-center justify-between px-[24px] py-[16px] border-b border-[#EEEEEE]">
            <div>
              <h2 className="font-bold text-[20px] text-[#13383B]">{chart.title}</h2>
              <p className="text-[12px] text-[#7E7E7E] mt-[2px]">{currentIdx + 1} of {allCharts.length} · Type: {chart.type}</p>
            </div>
            <div className="flex items-center gap-[8px]">
              <button onClick={() => setShowTypeSelector(!showTypeSelector)} className="px-[12px] py-[6px] bg-[#29ABB5] text-white text-[12px] rounded-[6px] hover:bg-[#238d96] transition flex items-center gap-[4px]"><PlusIcon className="w-[14px] h-[14px]"/>Convert</button>
              <button onClick={onClose} className="p-[8px] hover:bg-[#F5F5F5] rounded-[6px]"><XIcon className="w-[18px] h-[18px] text-[#575757]"/></button>
            </div>
          </div>
          {showTypeSelector && (
            <div className="px-[24px] py-[12px] bg-[#F5F5F5] border-b border-[#EEEEEE] flex items-center gap-[8px] overflow-x-auto" style={{ animation: 'slideDown 0.2s ease-out' }}>
              <span className="text-[12px] text-[#7E7E7E] flex-shrink-0">Convert to:</span>
              {CHART_TYPES.filter(t => t.id !== chart.type).map(type => {
                const TypeIcon = type.icon;
                return <button key={type.id} onClick={() => { onConvert(chart, type.id); setShowTypeSelector(false); }}
                  className="flex items-center gap-[4px] px-[10px] py-[5px] bg-white rounded-[6px] border border-[#EEEEEE] hover:border-[#29ABB5] transition text-[11px] text-[#365D60] flex-shrink-0"><TypeIcon className="w-[14px] h-[14px]"/>{type.label}</button>;
              })}
            </div>
          )}
          <div className="p-[24px]">
            <div className="bg-[#FAFAFA] rounded-[8px] p-[24px] min-h-[350px] flex items-center justify-center relative">
              <div className="w-full max-w-[600px]"><ChartRenderer chart={chart} onHover={setHovered}/></div>
              {hovered && (
                <div className="absolute top-[12px] right-[12px] bg-[#365D60] text-white px-[14px] py-[10px] rounded-[8px] shadow-lg z-10" style={{ animation: 'fadeIn 0.15s ease-out' }}>
                  <div className="font-bold text-[13px]">{hovered.label}</div>
                  <div className="text-[12px] opacity-90">{hovered.value?.toFixed(2)} mtCO2e</div>
                  {hovered.pct != null && <div className="text-[11px] opacity-70">{hovered.pct.toFixed(1)}% of total</div>}
                </div>
              )}
            </div>
            <div className="flex flex-wrap gap-[12px] mt-[16px]">
              {chart.data?.labels?.map((label, i) => (
                <div key={i} className="flex items-center gap-[6px]">
                  <div className="w-[10px] h-[10px] rounded-[2px]" style={{ backgroundColor: chart.colors[i % chart.colors.length] }}/>
                  <span className="text-[12px] text-[#575757]">{label}</span>
                  {chart.data.values && <span className="text-[11px] text-[#7E7E7E]">({chart.data.values[i]?.toFixed(0)})</span>}
                </div>
              ))}
            </div>
            <div className="mt-[16px] bg-[#E8F4F3] rounded-[8px] p-[14px]">
              <p className="text-[12px] font-bold text-[#365D60] mb-[6px]">Insights</p>
              {insights.map((ins, i) => <p key={i} className="text-[11px] text-[#575757] leading-[1.6]">• {ins}</p>)}
            </div>
            <div className="flex items-center justify-center gap-[6px] mt-[16px]">
              {allCharts.map(c => <button key={c.id} onClick={() => onNavigate(c)}
                className={`h-[8px] rounded-full transition-all duration-300 ${c.id === chart.id ? 'bg-[#29ABB5] w-[20px]' : 'bg-[#D9D9D9] hover:bg-[#ACC2C5] w-[8px]'}`}/>)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════
// DRAG & RESIZE HOOKS
// ═══════════════════════════════════════════════
const GRID_SNAP = 16;
const snap = (v, enabled) => enabled ? Math.round(v / GRID_SNAP) * GRID_SNAP : v;
const MIN_W = 280, MIN_H = 220;

function useDrag(canvasRef, position, onMove, snapEnabled) {
  const dragging = useRef(false);
  const offset = useRef({ x: 0, y: 0 });

  const onMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    dragging.current = true;
    const canvasRect = canvasRef.current?.getBoundingClientRect() || { left: 0, top: 0 };
    offset.current = {
      x: e.clientX - position.x - canvasRect.left + (canvasRef.current?.scrollLeft || 0),
      y: e.clientY - position.y - canvasRect.top + (canvasRef.current?.scrollTop || 0),
    };

    const onMouseMove = (e2) => {
      if (!dragging.current) return;
      const cr = canvasRef.current?.getBoundingClientRect() || { left: 0, top: 0 };
      const rawX = e2.clientX - offset.current.x - cr.left + (canvasRef.current?.scrollLeft || 0);
      const rawY = e2.clientY - offset.current.y - cr.top + (canvasRef.current?.scrollTop || 0);
      onMove({ x: snap(Math.max(0, rawX), snapEnabled), y: snap(Math.max(0, rawY), snapEnabled) });
    };
    const onMouseUp = () => {
      dragging.current = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [canvasRef, position, onMove, snapEnabled]);

  return onMouseDown;
}

function useResize(canvasRef, position, size, onResize, snapEnabled) {
  const resizing = useRef(false);

  const onMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    resizing.current = true;
    const startX = e.clientX, startY = e.clientY;
    const startW = size.w, startH = size.h;

    const onMouseMove = (e2) => {
      if (!resizing.current) return;
      const newW = snap(Math.max(MIN_W, startW + (e2.clientX - startX)), snapEnabled);
      const newH = snap(Math.max(MIN_H, startH + (e2.clientY - startY)), snapEnabled);
      onResize({ w: newW, h: newH });
    };
    const onMouseUp = () => {
      resizing.current = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [size, onResize, snapEnabled]);

  return onMouseDown;
}

// ═══════════════════════════════════════════════
// Draggable Chart Card (Canvas item)
// ═══════════════════════════════════════════════
function DraggableChartCard({ chart, layout, canvasRef, snapEnabled, locked,
  onUpdateLayout, onBringToFront, onClickChart, onConvertClick, onDelete }) {
  const [hovered, setHovered] = useState(null);
  const [isHover, setIsHover] = useState(false);

  const onMove = useCallback((pos) => onUpdateLayout(chart.id, { ...layout, x: pos.x, y: pos.y }), [chart.id, layout, onUpdateLayout]);
  const onResize = useCallback((sz) => onUpdateLayout(chart.id, { ...layout, w: sz.w, h: sz.h }), [chart.id, layout, onUpdateLayout]);
  const dragHandler = useDrag(canvasRef, layout, onMove, snapEnabled);
  const resizeHandler = useResize(canvasRef, layout, layout, onResize, snapEnabled);

  return (
    <div
      className="absolute bg-white rounded-[10px] border border-[#EEEEEE] shadow-sm hover:shadow-xl transition-shadow duration-200 flex flex-col select-none"
      style={{
        left: layout.x, top: layout.y, width: layout.w, height: layout.h,
        zIndex: layout.z || 1,
        animation: layout.isNew ? 'chartDrop 0.35s ease-out' : undefined,
      }}
      onMouseDown={() => onBringToFront(chart.id)}
      onMouseEnter={() => setIsHover(true)}
      onMouseLeave={() => { setIsHover(false); setHovered(null); }}>

      {/* Header — drag handle */}
      <div
        className={`flex items-center gap-[8px] px-[14px] py-[10px] border-b border-[#F0F0F0] rounded-t-[10px] ${locked ? 'cursor-default' : 'cursor-grab active:cursor-grabbing'}`}
        style={{ background: 'linear-gradient(to right, #FAFAFA, #F5F5F5)' }}
        onMouseDown={locked ? undefined : dragHandler}>
        
        {!locked && <MoveIcon className="w-[12px] h-[12px] text-[#ACC2C5] flex-shrink-0"/>}
        <h3 className="font-['Noto_Sans',sans-serif] font-bold text-[13px] text-[#13383B] truncate flex-1">{chart.title}</h3>

        <div className={`flex items-center gap-[2px] transition-opacity duration-200 ${isHover ? 'opacity-100' : 'opacity-0'}`}>
          <button onClick={(e) => { e.stopPropagation(); onConvertClick(chart); }} className="p-[4px] hover:bg-[#E8F4F3] rounded-[4px]" title="Generate variant">
            <PlusIcon className="w-[14px] h-[14px] text-[#29ABB5]"/>
          </button>
          <button onClick={(e) => { e.stopPropagation(); onClickChart(chart); }} className="p-[4px] hover:bg-[#E8F4F3] rounded-[4px]" title="Expand">
            <MaximizeIcon className="w-[13px] h-[13px] text-[#365D60]"/>
          </button>
          {chart.isGenerated && (
            <button onClick={(e) => { e.stopPropagation(); onDelete(chart.id); }} className="p-[4px] hover:bg-red-50 rounded-[4px]" title="Remove">
              <TrashIcon className="w-[13px] h-[13px] text-red-400"/>
            </button>
          )}
        </div>
      </div>

      {/* Chart body */}
      <div className="flex-1 p-[10px] overflow-hidden relative cursor-pointer"
        onClick={() => onClickChart(chart)}>
        <ChartRenderer chart={chart} onHover={setHovered}/>
        {hovered && (
          <div className="absolute top-[6px] right-[6px] bg-[#365D60] text-white px-[10px] py-[6px] rounded-[6px] text-[10px] z-10 shadow-lg pointer-events-none" style={{ animation: 'fadeIn 0.1s ease-out' }}>
            <div className="font-bold">{hovered.label}</div>
            <div>{hovered.value?.toFixed(1)} mtCO2e</div>
            {hovered.pct != null && <div className="opacity-70">{hovered.pct.toFixed(1)}%</div>}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="px-[10px] pb-[8px] flex flex-wrap gap-[6px]">
        {chart.data?.labels?.slice(0, 5).map((label, i) => (
          <div key={i} className="flex items-center gap-[3px]">
            <div className="w-[7px] h-[7px] rounded-[1px]" style={{ backgroundColor: chart.colors[i % chart.colors.length] }}/>
            <span className="text-[9px] text-[#7E7E7E]">{label}</span>
          </div>
        ))}
      </div>

      {/* Resize handle (bottom-right) */}
      {!locked && (
        <div className="absolute bottom-0 right-0 w-[18px] h-[18px] cursor-nwse-resize group z-10" onMouseDown={resizeHandler}>
          <svg viewBox="0 0 18 18" className="w-full h-full">
            <path d="M14 4L4 14M14 8L8 14M14 12L12 14" stroke={isHover ? '#29ABB5' : '#D9D9D9'} strokeWidth="1.5" strokeLinecap="round" className="transition-colors"/>
          </svg>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════
// Canvas Toolbar
// ═══════════════════════════════════════════════
function CanvasToolbar({ snapEnabled, setSnapEnabled, locked, setLocked, onAutoArrange, chartCount }) {
  return (
    <div className="flex items-center gap-[6px] mb-[12px]">
      <div className="flex items-center bg-[#F5F5F5] rounded-[8px] p-[3px] gap-[2px]">
        <button onClick={() => setSnapEnabled(!snapEnabled)} title={snapEnabled ? 'Snap to grid ON' : 'Snap to grid OFF'}
          className={`flex items-center gap-[4px] px-[10px] py-[5px] rounded-[6px] text-[11px] transition-all duration-200 ${snapEnabled ? 'bg-white shadow-sm text-[#365D60] font-medium' : 'text-[#7E7E7E] hover:text-[#365D60]'}`}>
          <GridIcon className="w-[13px] h-[13px]"/>Snap
        </button>
        <button onClick={() => setLocked(!locked)} title={locked ? 'Unlock layout' : 'Lock layout'}
          className={`flex items-center gap-[4px] px-[10px] py-[5px] rounded-[6px] text-[11px] transition-all duration-200 ${locked ? 'bg-white shadow-sm text-[#C2185B] font-medium' : 'text-[#7E7E7E] hover:text-[#365D60]'}`}>
          {locked ? <LockIcon className="w-[13px] h-[13px]"/> : <UnlockIcon className="w-[13px] h-[13px]"/>}
          {locked ? 'Locked' : 'Free'}
        </button>
        <button onClick={onAutoArrange} title="Auto-arrange charts"
          className="flex items-center gap-[4px] px-[10px] py-[5px] rounded-[6px] text-[11px] text-[#7E7E7E] hover:text-[#365D60] hover:bg-white transition-all duration-200">
          <LayoutIcon className="w-[13px] h-[13px]"/>Auto
        </button>
      </div>
      <span className="text-[11px] text-[#ACC2C5] ml-[8px]">{chartCount} chart{chartCount !== 1 ? 's' : ''} on canvas</span>
    </div>
  );
}

// ═══════════════════════════════════════════════
// Main Dashboard Component
// ═══════════════════════════════════════════════
export default function Dashboard({ onBack, sessionId, calculationResult, onGoToStep, onLogoClick }) {
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({ facilities: [], activityTypes: [], year: 'All' });
  const [selectedChart, setSelectedChart] = useState(null);
  const [convertSource, setConvertSource] = useState(null);
  const [allChartsState, setAllChartsState] = useState([]);
  const [layouts, setLayouts] = useState({});
  const [nextId, setNextId] = useState(100);
  const [topZ, setTopZ] = useState(10);
  const [snapEnabled, setSnapEnabled] = useState(true);
  const [locked, setLocked] = useState(false);
  const canvasRef = useRef(null);

  const [apiData, setApiData] = useState(null);
  const [dataLoading, setDataLoading] = useState(false);

  // Fetch live analytics from backend; fall back to DASHBOARD_DATA if unavailable
  useEffect(() => {
    if (!sessionId) return;
    setDataLoading(true);
    Promise.all([
      getAnalyticsSummary(sessionId),
      getChartData(sessionId, 'stacked_bar', 'mtCO2e_calc', 'Year', []).catch(() => null),
    ])
      .then(([summaryRes, stackedRes]) => {
        const s = summaryRes.summary;

        // Build bySourceYear from stacked_bar response datasets
        let bySourceYear = DASHBOARD_DATA.bySourceYear;
        if (stackedRes?.datasets?.length) {
          bySourceYear = {};
          stackedRes.datasets.forEach(ds => {
            bySourceYear[ds.label] = {};
            (stackedRes.labels || []).forEach((yr, i) => {
              bySourceYear[ds.label][String(yr)] = ds.data[i] || 0;
            });
          });
        }

        // Build byCommodityYear from bySubtype flat breakdown per year
        const byCommodityYear = {};
        if (s.byYear) {
          Object.keys(s.byYear).forEach(yr => {
            byCommodityYear[yr] = s.bySubtype || {};
          });
        }

        const sector = s.bySector || {};
        setApiData({
          byYear: s.byYear || DASHBOARD_DATA.byYear,
          bySourceYear,
          byCommodityYear,
          totals: {
            Fleet:    sector.Fleet    || sector.fleet    || 0,
            PSE:      sector.PSE      || sector.Facilities || 0,
            SCL:      sector.SCL      || 0,
            Landfill: sector.Landfill || sector.landfill  || 0,
            Total:    s.totalMtCO2e   || 0,
          },
        });
        setDataLoading(false);
      })
      .catch(err => {
        console.error('[Dashboard] Failed to load live analytics, using mock data:', err);
        setDataLoading(false);
      });
  }, [sessionId]);

  const data = apiData || DASHBOARD_DATA;

  // Filter
  const filteredData = useMemo(() => {
    let sources = { ...data.bySourceYear };
    if (filters.facilities.length > 0) {
      const fs = filters.facilities.flatMap(f => f === 'PSE' ? ['PSE'] : f === 'SCL' ? ['SCL'] : []);
      if (fs.length > 0) sources = Object.fromEntries(Object.entries(sources).filter(([k]) => fs.includes(k)));
    }
    if (filters.activityTypes.length > 0) {
      const allowed = new Set();
      filters.activityTypes.forEach(at => { (ACTIVITY_MAPPING[at] || []).forEach(s => allowed.add(s)); });
      if (allowed.size > 0) sources = Object.fromEntries(Object.entries(sources).filter(([k]) => allowed.has(k)));
    }
    const years = ['2019', '2020', '2021', '2022'];
    const fy = filters.year !== 'All' ? [filters.year] : years;
    const byYear = {}; fy.forEach(y => { byYear[y] = Object.values(sources).reduce((s, src) => s + (src[y] || 0), 0); });
    const bySource = {}; Object.entries(sources).forEach(([src, yd]) => { bySource[src] = fy.reduce((s, y) => s + (yd[y] || 0), 0); });
    const total = Object.values(bySource).reduce((s, v) => s + v, 0);
    return { byYear, bySource, sources, total, filteredYears: fy };
  }, [data, filters]);

  // Build base charts
  const baseCharts = useMemo(() => {
    const { byYear, bySource, sources, filteredYears } = filteredData;
    return [
      { id: 'chart-1', title: 'GHG Emissions by Year (Stacked)', type: 'stacked-bar', colors: COLOR_PALETTES.teal,
        data: { labels: filteredYears, values: filteredYears.map(y => byYear[y] || 0), stackData: sources } },
      { id: 'chart-2', title: 'Emissions by Source', type: 'pie', colors: COLOR_PALETTES.purple,
        data: { labels: Object.keys(bySource), values: Object.values(bySource) } },
      { id: 'chart-3', title: 'Emissions Trend Over Time', type: 'area', colors: COLOR_PALETTES.gold,
        data: { labels: filteredYears, values: filteredYears.map(y => byYear[y] || 0) } },
      { id: 'chart-4', title: 'Source Comparison', type: 'hbar', colors: COLOR_PALETTES.rose,
        data: { labels: Object.keys(bySource), values: Object.values(bySource) } },
    ];
  }, [filteredData]);

  // Initialize layouts for base charts (2×2 grid)
  useEffect(() => {
    if (Object.keys(layouts).length === 0) {
      const gap = 20, colW = 420, rowH = 340;
      const initial = {};
      baseCharts.forEach((c, i) => {
        const col = i % 2, row = Math.floor(i / 2);
        initial[c.id] = { x: col * (colW + gap), y: row * (rowH + gap), w: colW, h: rowH, z: i + 1 };
      });
      setLayouts(initial);
      setAllChartsState(baseCharts);
    } else {
      // Update chart data without changing positions
      setAllChartsState(prev => {
        const genCharts = prev.filter(c => c.isGenerated);
        return [...baseCharts, ...genCharts];
      });
    }
  }, [baseCharts, layouts]);

  const allCharts = allChartsState;

  // Compute canvas dimensions from chart positions
  const canvasSize = useMemo(() => {
    let maxX = 900, maxY = 700;
    Object.values(layouts).forEach(l => {
      maxX = Math.max(maxX, l.x + l.w + 40);
      maxY = Math.max(maxY, l.y + l.h + 40);
    });
    return { width: maxX, height: maxY };
  }, [layouts]);

  const handleConvert = useCallback((sourceChart, newType) => {
    const palettes = [COLOR_PALETTES.teal, COLOR_PALETTES.gold, COLOR_PALETTES.purple, COLOR_PALETTES.rose];
    const typeName = CHART_TYPES.find(t => t.id === newType)?.label || newType;
    const newId = `gen-${nextId}`;
    const newChart = {
      id: newId,
      title: `${sourceChart.title.replace(/ \(.*\)$/, '')} (${typeName})`,
      type: newType,
      colors: palettes[nextId % palettes.length],
      data: { ...sourceChart.data },
      isGenerated: true,
    };
    // Place near parent chart
    const parentLayout = layouts[sourceChart.id] || { x: 0, y: 0, w: 400, h: 320 };
    const newZ = topZ + 1;
    setTopZ(newZ);
    setLayouts(prev => ({
      ...prev,
      [newId]: {
        x: snap(parentLayout.x + 40, snapEnabled),
        y: snap(parentLayout.y + 40, snapEnabled),
        w: parentLayout.w,
        h: parentLayout.h,
        z: newZ,
        isNew: true,
      }
    }));
    setAllChartsState(prev => [...prev, newChart]);
    setNextId(p => p + 1);
    // Clear "isNew" after animation
    setTimeout(() => {
      setLayouts(prev => prev[newId] ? { ...prev, [newId]: { ...prev[newId], isNew: false } } : prev);
    }, 400);
  }, [nextId, layouts, topZ, snapEnabled]);

  const handleDelete = useCallback((chartId) => {
    setAllChartsState(prev => prev.filter(c => c.id !== chartId));
    setLayouts(prev => { const n = { ...prev }; delete n[chartId]; return n; });
    if (selectedChart?.id === chartId) setSelectedChart(null);
  }, [selectedChart]);

  const updateLayout = useCallback((id, newLayout) => {
    setLayouts(prev => ({ ...prev, [id]: { ...prev[id], ...newLayout } }));
  }, []);

  const bringToFront = useCallback((id) => {
    const nz = topZ + 1;
    setTopZ(nz);
    setLayouts(prev => ({ ...prev, [id]: { ...prev[id], z: nz } }));
  }, [topZ]);

  const autoArrange = useCallback(() => {
    const gap = 20, colW = 420, rowH = 340, cols = 2;
    const newLayouts = {};
    allCharts.forEach((c, i) => {
      const col = i % cols, row = Math.floor(i / cols);
      newLayouts[c.id] = { x: col * (colW + gap), y: row * (rowH + gap), w: colW, h: rowH, z: i + 1 };
    });
    setLayouts(newLayouts);
    setTopZ(allCharts.length + 1);
  }, [allCharts]);

  const toggleFilter = (type, value) => {
    setFilters(prev => {
      if (type === 'facility') { const n = prev.facilities.includes(value) ? prev.facilities.filter(f => f !== value) : [...prev.facilities, value]; return { ...prev, facilities: n }; }
      if (type === 'activity') { const n = prev.activityTypes.includes(value) ? prev.activityTypes.filter(a => a !== value) : [...prev.activityTypes, value]; return { ...prev, activityTypes: n }; }
      if (type === 'year') return { ...prev, year: value };
      return prev;
    });
  };

  const activeFiltersCount = filters.facilities.length + filters.activityTypes.length + (filters.year !== 'All' ? 1 : 0);
  const handleLogoClick = () => {
    if (onLogoClick) onLogoClick();
  };
  const completedSteps = [
    { label: 'Upload', step: 1 },
    { label: 'Cleaning', step: 3 },
    { label: 'Processing', step: 5 },
  ];

  // Detail view
  if (selectedChart) {
    return <ChartDetailView chart={selectedChart} allCharts={allCharts}
      onClose={() => setSelectedChart(null)} onNavigate={setSelectedChart} onConvert={handleConvert}/>;
  }

  return (
    <div className="bg-white w-full min-h-screen flex flex-col">
      {/* Live data loading indicator */}
      {dataLoading && (
        <div className="fixed top-[62px] left-0 right-0 z-50 bg-[#29ABB5]/10 border-b border-[#29ABB5]/20 text-center py-[6px]">
          <span className="font-['Noto_Sans',sans-serif] text-[12px] text-[#29ABB5] font-medium">
            Loading live analytics data...
          </span>
        </div>
      )}
      {/* Header */}
      <div className="sticky top-0 left-0 w-full h-[62px] bg-[#365D60] flex items-center justify-between px-[64px] z-20">
        <button
          type="button"
          onClick={handleLogoClick}
          className="flex items-center gap-[12px] bg-transparent border-0 p-0 cursor-pointer"
        >
          <img src="/logo.png" alt="Logo" className="h-[40px] w-auto"/>
          <p className="font-['Noto_Sans',sans-serif] font-bold text-[24px] text-white">EcoMetrics</p>
        </button>
        <div className="bg-white rounded-[10px] w-[331px] h-[40px] flex items-center px-[12px]">
          <SearchIcon className="w-[20px] h-[20px] text-[#7E7E7E]"/>
        </div>
      </div>

      {/* Progress Steps + right-side icons */}
      <div className="sticky top-[62px] bg-white py-[19px] px-[64px] flex items-center z-10 border-b border-[#EEEEEE]">
        {completedSteps.map((stepItem) => (
          <React.Fragment key={stepItem.step}>
            <div
              className="flex items-center gap-[12px] cursor-pointer"
              onClick={() => onGoToStep && onGoToStep(stepItem.step)}
            >
              <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center"><p className="font-bold text-[14px] text-white">✓</p></div>
              <p className="font-normal text-[14px] text-[#13383B]">{stepItem.label}</p>
            </div>
            <div className="flex-1 h-[2px] bg-[#50B461] mx-[16px]"></div>
          </React.Fragment>
        ))}
        <div className="flex items-center gap-[12px]">
          <div className="w-[32px] h-[32px] rounded-full bg-[#29ABB5] flex items-center justify-center"><p className="font-bold text-[14px] text-white">4</p></div>
          <p className="font-normal text-[14px] text-[#13383B]">Visualization</p>
        </div>
        <div className="ml-auto flex items-center gap-[6px] pl-[24px]">
          <button onClick={() => setShowFilters(prev => !prev)}
            className={`relative p-[8px] rounded-[8px] transition-all duration-200 cursor-pointer ${showFilters ? 'bg-[#29ABB5] text-white shadow-sm' : 'hover:bg-[#F0F0F0] text-[#575757]'}`} title="Toggle filters">
            <FilterIcon className="w-[18px] h-[18px]"/>
            {activeFiltersCount > 0 && !showFilters && (
              <span className="absolute -top-[4px] -right-[4px] w-[16px] h-[16px] bg-[#C2185B] text-white text-[9px] font-bold rounded-full flex items-center justify-center">{activeFiltersCount}</span>
            )}
          </button>
          <button className="p-[8px] rounded-[8px] hover:bg-[#F0F0F0] text-[#575757] transition cursor-pointer" title="Grid view"><GridIcon className="w-[18px] h-[18px]"/></button>
        </div>
      </div>

      {/* Filter Bar — smooth slide */}
      <div style={{
        maxHeight: showFilters ? '80px' : '0px', opacity: showFilters ? 1 : 0, overflow: 'hidden',
        transition: 'max-height 0.4s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease-in-out',
      }}>
        <div className="bg-[#365D60] px-[64px] py-[12px] flex items-center gap-[20px]">
          <span className="text-white text-[13px] font-medium flex-shrink-0">Facilities</span>
          {['Seattle City Light', 'Puget Sound Energy'].map(name => {
            const key = name === 'Seattle City Light' ? 'SCL' : 'PSE';
            const active = filters.facilities.includes(key);
            return <button key={key} onClick={() => toggleFilter('facility', key)}
              className={`px-[14px] py-[5px] text-[12px] rounded-[6px] border transition-all duration-200 flex-shrink-0 ${active ? 'bg-[#29ABB5] text-white border-[#29ABB5]' : 'bg-white/10 text-white border-white/30 hover:bg-white/20'}`}>{name}</button>;
          })}
          <div className="w-[1px] h-[24px] bg-white/20 flex-shrink-0"/>
          <span className="text-white text-[13px] font-medium flex-shrink-0">Activity Type</span>
          {['Fossil Fuel', 'Gas', 'Electricity', 'Landfill', 'Fleet'].map(type => {
            const active = filters.activityTypes.includes(type);
            return <button key={type} onClick={() => toggleFilter('activity', type)}
              className={`px-[14px] py-[5px] text-[12px] rounded-[6px] border transition-all duration-200 flex-shrink-0 ${active ? 'bg-[#29ABB5] text-white border-[#29ABB5]' : 'bg-white/10 text-white border-white/30 hover:bg-white/20'}`}>{type}</button>;
          })}
          <div className="w-[1px] h-[24px] bg-white/20 flex-shrink-0"/>
          <span className="text-white text-[13px] font-medium flex-shrink-0">Year</span>
          <select value={filters.year} onChange={e => toggleFilter('year', e.target.value)}
            className="bg-[#29ABB5] text-white text-[12px] rounded-[6px] px-[12px] py-[5px] border-none cursor-pointer outline-none flex-shrink-0">
            <option value="All">All</option>
            {['2019','2020','2021','2022'].map(y => <option key={y} value={y}>{y}</option>)}
          </select>
          {activeFiltersCount > 0 && (
            <button onClick={() => setFilters({ facilities: [], activityTypes: [], year: 'All' })}
              className="ml-auto text-[11px] text-white/70 hover:text-white transition px-[8px] py-[4px] rounded-[4px] hover:bg-white/10 flex-shrink-0">Clear all ({activeFiltersCount})</button>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 px-[64px] py-[24px]">
        <OverallEmissionsBanner data={data}/>

        {/* Canvas Toolbar */}
        <CanvasToolbar snapEnabled={snapEnabled} setSnapEnabled={setSnapEnabled} locked={locked} setLocked={setLocked} onAutoArrange={autoArrange} chartCount={allCharts.length}/>

        {/* ═══ FREE-FORM CANVAS ═══ */}
        <div ref={canvasRef} className="relative overflow-auto rounded-[12px] border-2 border-dashed border-[#E0E0E0]"
          style={{
            width: '100%',
            height: `${Math.max(canvasSize.height, 720)}px`,
            background: snapEnabled
              ? `radial-gradient(circle, #E8E8E8 1px, transparent 1px)`
              : '#FAFAFA',
            backgroundSize: snapEnabled ? `${GRID_SNAP}px ${GRID_SNAP}px` : undefined,
          }}>
          {allCharts.map(chart => {
            const layout = layouts[chart.id];
            if (!layout) return null;
            return (
              <DraggableChartCard
                key={chart.id}
                chart={chart}
                layout={layout}
                canvasRef={canvasRef}
                snapEnabled={snapEnabled}
                locked={locked}
                onUpdateLayout={updateLayout}
                onBringToFront={bringToFront}
                onClickChart={setSelectedChart}
                onConvertClick={setConvertSource}
                onDelete={handleDelete}
              />
            );
          })}
          {allCharts.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center text-[#ACC2C5] text-[14px]">
              No charts to display — adjust filters or generate new charts
            </div>
          )}
        </div>

        <button onClick={onBack}
          className="mt-[24px] px-[32px] py-[12px] bg-[#EEEEEE] rounded-[24.5px] font-medium text-[16px] text-[#575757] hover:bg-[#D9D9D9] transition">
          ← Back to Results
        </button>
      </div>

      {/* Convert overlay */}
      {convertSource && <ChartConvertOverlay chart={convertSource} onGenerate={handleConvert} onClose={() => setConvertSource(null)}/>}

      <style>{`
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes slideDown { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeSlide { from { opacity: 0; transform: translateX(10px); } to { opacity: 1; transform: translateX(0); } }
        @keyframes cardIn { from { opacity: 0; transform: scale(0.97); } to { opacity: 1; transform: scale(1); } }
        @keyframes chartDrop { from { opacity: 0; transform: scale(0.85) translateY(-20px); } to { opacity: 1; transform: scale(1) translateY(0); } }
      `}</style>
    </div>
  );
}
