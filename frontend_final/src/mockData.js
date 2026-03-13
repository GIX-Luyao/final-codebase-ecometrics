// ═══════════════════════════════════════════════════════════
// 📦 模拟原始数据（Preview Files 使用）
// ═══════════════════════════════════════════════════════════
export const MOCK_RAW_DATA = {
  fleet: [
    { Consumption: 25.5, Unit: 'gal', Subtype: 'Gasoline', Year: 2021, Month: 10, LowOrg: 'SU092', Vehicle_Location: 'OCC-SPU', Vehicle_Class: 'Light', FillDate: '2021-10-30' },
    { Consumption: 45.2, Unit: 'gal', Subtype: 'B20', Year: 2021, Month: 10, LowOrg: 'SU093', Vehicle_Location: 'WSC-SPU', Vehicle_Class: 'Heavy', FillDate: '2021-10-30' },
    { Consumption: 18.3, Unit: 'gal', Subtype: 'Gasoline', Year: 2021, Month: 11, LowOrg: 'SU092', Vehicle_Location: 'OCC-SPU', Vehicle_Class: 'Light', FillDate: '2021-11-15' },
    { Consumption: 52.7, Unit: 'gal', Subtype: 'Diesel', Year: 2022, Month: 1, LowOrg: 'SU094', Vehicle_Location: 'EAB-SPU', Vehicle_Class: 'Heavy', FillDate: '2022-01-10' },
    { Consumption: 32.1, Unit: 'gal', Subtype: 'Gasoline', Year: 2022, Month: 2, LowOrg: 'SU095', Vehicle_Location: 'NCC-SPU', Vehicle_Class: 'Light', FillDate: '2022-02-15' },
  ],
  pse: [
    { ACCT_ID: '200003770308', Utility: 'PSE', Year: 2022, Consumption: 7918489.562, Unit: 'KWH' },
    { ACCT_ID: '200003770309', Utility: 'PSE', Year: 2021, Consumption: 8200000, Unit: 'KWH' },
    { ACCT_ID: '200003770310', Utility: 'PSE', Year: 2020, Consumption: 7500000, Unit: 'KWH' },
  ],
  scl: [
    { ACCT_ID: '10710000', Utility: 'SCL', Year: 2022, Consumption: 694130.793, Unit: 'KWH' },
    { ACCT_ID: '10710001', Utility: 'SCL', Year: 2021, Consumption: 720000, Unit: 'KWH' },
  ],
  landfill: [
    { Year: 2022, Consumption: 334.51, Unit: 'mt CH4', Subtype: 'Landfill Methane', Location: 'Cedar Hills' },
    { Year: 2021, Consumption: 310.2, Unit: 'mt CH4', Subtype: 'Landfill Methane', Location: 'Cedar Hills' },
    { Year: 2020, Consumption: 295.8, Unit: 'mt CH4', Subtype: 'Landfill Methane', Location: 'Cedar Hills' },
  ]
};

// ═══════════════════════════════════════════════════════════
// 📊 模拟计算后的数据（Processed Results 使用）
// ═══════════════════════════════════════════════════════════
export const MOCK_CALCULATED_DATA = [
  { 
    Consumption: 25.5, Unit: 'gal', Subtype: 'Gasoline', GHG: 'CO2', Year: 2021, Month: 10, LowOrg: 'SU092',
    EF_ID: 'Fuel_Gasoline_2021_CO2_gal', GHG_MTperUnit: 0.008887, GWP: 1, mtCO2e_calc: 0.226618,
    formula: 'mtCO2e = Consumption × GHG_MTperUnit × GWP = 25.5 × 0.008887 × 1 = 0.226618'
  },
  { 
    Consumption: 45.2, Unit: 'gal', Subtype: 'B20', GHG: 'CO2', Year: 2021, Month: 10, LowOrg: 'SU093',
    EF_ID: 'Fuel_B20_2021_CO2_gal', GHG_MTperUnit: 0.008211, GWP: 1, mtCO2e_calc: 0.371137,
    formula: 'mtCO2e = Consumption × GHG_MTperUnit × GWP = 45.2 × 0.008211 × 1 = 0.371137'
  },
  { 
    Consumption: 18.3, Unit: 'gal', Subtype: 'Gasoline', GHG: 'CO2', Year: 2021, Month: 11, LowOrg: 'SU092',
    EF_ID: 'Fuel_Gasoline_2021_CO2_gal', GHG_MTperUnit: 0.008887, GWP: 1, mtCO2e_calc: 0.162632,
    formula: 'mtCO2e = Consumption × GHG_MTperUnit × GWP = 18.3 × 0.008887 × 1 = 0.162632'
  },
  { 
    Consumption: 52.7, Unit: 'gal', Subtype: 'Diesel', GHG: 'CO2', Year: 2022, Month: 1, LowOrg: 'SU094',
    EF_ID: 'Fuel_Diesel_2022_CO2_gal', GHG_MTperUnit: 0.01018, GWP: 1, mtCO2e_calc: 0.536486,
    formula: 'mtCO2e = Consumption × GHG_MTperUnit × GWP = 52.7 × 0.01018 × 1 = 0.536486'
  },
  { 
    Consumption: 32.1, Unit: 'gal', Subtype: 'Gasoline', GHG: 'CO2', Year: 2022, Month: 2, LowOrg: 'SU095',
    EF_ID: 'Fuel_Gasoline_2022_CO2_gal', GHG_MTperUnit: 0.008887, GWP: 1, mtCO2e_calc: 0.285327,
    formula: 'mtCO2e = Consumption × GHG_MTperUnit × GWP = 32.1 × 0.008887 × 1 = 0.285327'
  },
  {
    ACCT_ID: '200003770308', Utility: 'PSE', Year: 2022, Consumption: 7918489.562, Unit: 'KWH', GHG: 'CO2',
    EF_ID: 'Elec_PSE_2022_CO2', GHG_MTperUnit: 0.00041595, GWP: 1, mtCO2e_calc: 3293.654,
    formula: 'mtCO2e = Consumption × GHG_MTperUnit × GWP = 7918489.562 × 0.00041595 × 1 = 3293.654'
  },
  {
    ACCT_ID: '200003770309', Utility: 'PSE', Year: 2021, Consumption: 8200000, Unit: 'KWH', GHG: 'CO2',
    EF_ID: 'Elec_PSE_2021_CO2', GHG_MTperUnit: 0.000416, GWP: 1, mtCO2e_calc: 3411.2,
    formula: 'mtCO2e = Consumption × GHG_MTperUnit × GWP = 8200000 × 0.000416 × 1 = 3411.2'
  },
  {
    ACCT_ID: '10710000', Utility: 'SCL', Year: 2022, Consumption: 694130.793, Unit: 'KWH', GHG: 'CO2',
    EF_ID: 'Elec_SCL_2022_CO2', GHG_MTperUnit: 0.00001051, GWP: 1, mtCO2e_calc: 7.295,
    formula: 'mtCO2e = Consumption × GHG_MTperUnit × GWP = 694130.793 × 0.00001051 × 1 = 7.295'
  },
  {
    Year: 2022, Consumption: 334.51, Unit: 'mt CH4', Subtype: 'Landfill Methane', GHG: 'CH4',
    EF_ID: 'Landfill_CH4', GHG_MTperUnit: 1.0, GWP: 28, mtCO2e_calc: 9366.28,
    formula: 'mtCO2e = Consumption × GHG_MTperUnit × GWP = 334.51 × 1.0 × 28 = 9366.28'
  },
  {
    Year: 2021, Consumption: 310.2, Unit: 'mt CH4', Subtype: 'Landfill Methane', GHG: 'CH4',
    EF_ID: 'Landfill_CH4', GHG_MTperUnit: 1.0, GWP: 28, mtCO2e_calc: 8685.6,
    formula: 'mtCO2e = Consumption × GHG_MTperUnit × GWP = 310.2 × 1.0 × 28 = 8685.6'
  },
];

// ═══════════════════════════════════════════════════════════
// 📝 字段说明（PreviewFiles 悬停提示）
// ═══════════════════════════════════════════════════════════
export const FIELD_DESCRIPTIONS = {
  // Fleet fields
  Consumption: '燃料消耗量 - 车辆加油的数量',
  Unit: '单位 - 燃料的计量单位（加仑、升等）',
  Subtype: '燃料类型 - 使用的燃料种类（汽油、柴油、B20等）',
  Year: '年份 - 数据记录的年份',
  Month: '月份 - 数据记录的月份',
  LowOrg: '组织代码 - 低级组织单位标识符',
  Vehicle_Location: '车辆位置 - 车辆所属的地点或部门',
  Vehicle_Class: '车辆类别 - 车辆的分类（轻型、重型等）',
  FillDate: '加油日期 - 车辆加油的具体日期',
  
  // PSE/SCL fields
  ACCT_ID: '账户ID - 电力账户的唯一标识符',
  Utility: '公用事业公司 - 提供电力服务的公司名称（PSE或SCL）',
  Location: '位置 - 设施或数据源的地理位置',
  
  // Calculated fields
  GHG: '温室气体类型 - CO2, CH4, N2O等',
  EF_ID: '排放因子ID - 用于计算的排放因子标识符',
  GHG_MTperUnit: '排放因子 - 每单位消耗的温室气体排放量（公吨CO2e）',
  GWP: '全球变暖潜势 - 相对于CO2的温室效应倍数',
  mtCO2e_calc: '计算结果 - 以公吨CO2当量表示的总排放量',
};

// ═══════════════════════════════════════════════════════════
// 📈 Dashboard 图表数据
// ═══════════════════════════════════════════════════════════
export const DASHBOARD_CHARTS = {
  summary: {
    totalMtCO2e: 16068.46,
    recordCount: 10,
  },
  charts: {
    // Pie chart data
    pieChart: {
      data: [
        { label: 'Fossil Fuel', value: 1.582, percentage: 45 },
        { label: 'Electricity', value: 6712.15, percentage: 25 },
        { label: 'Waste', value: 18051.88, percentage: 20 },
        { label: 'Refrigerants', value: 0.52, percentage: 10 }
      ]
    },
    // Bar chart data
    barChart: {
      labels: ['2019', '2020', '2021', '2022'],
      datasets: [{
        data: [2871.8, 3120.2, 11909.4, 12770.18]
      }]
    }
  }
};
