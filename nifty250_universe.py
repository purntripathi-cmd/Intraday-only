"""
Full NIFTY LargeMidcap 250 Universe + Sectoral & International ETFs (`nifty250_universe.py`)
===========================================================================================
Provides:
  - `NIFTY_250_RAW`: All 250 liquid NIFTY LargeMidcap 250 stocks (`(ticker, name, category)`).
  - `CORE_54_UNIVERSE_META`: Original 54-instrument core backtest dictionary.
  - `UNIVERSE_META`: Complete 261-instrument dictionary (250 NIFTY 250 stocks + 11 Indian/International ETFs)
    mapped to NSE Sectoral Indices (`^NSEBANK`, `^CNXIT`, `^CNXPHARMA`, `^CNXAUTO`, `^CNXMETAL`,
    `^CNXENERGY`, `^CNXFMCG`, `^NSEI`) and International Benchmarks (`^IXIC`, `^GSPC`, `GC=F`).
"""

CORE_54_UNIVERSE_META = {
    # Pharma & Healthcare
    "DIVISLAB.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    "SUNPHARMA.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    "CIPLA.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    "DRREDDY.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    "AJANTPHARM.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    "LAURUSLABS.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    "AUROPHARMA.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    "PHARMABEES.NS": {"sector": "^CNXPHARMA", "group": "PHARMA_DEFENSIVE", "market": "INDIA_NSE"},
    # IT & Global Tech Exporters
    "TCS.NS": {"sector": "^CNXIT", "group": "IT_GLOBAL", "market": "INDIA_NSE"},
    "INFY.NS": {"sector": "^CNXIT", "group": "IT_GLOBAL", "market": "INDIA_NSE"},
    "HCLTECH.NS": {"sector": "^CNXIT", "group": "IT_GLOBAL", "market": "INDIA_NSE"},
    "TECHM.NS": {"sector": "^CNXIT", "group": "IT_GLOBAL", "market": "INDIA_NSE"},
    "PERSISTENT.NS": {"sector": "^CNXIT", "group": "IT_GLOBAL", "market": "INDIA_NSE"},
    "ITBEES.NS": {"sector": "^CNXIT", "group": "IT_GLOBAL", "market": "INDIA_NSE"},
    # Banking & Financials
    "HDFCBANK.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "ICICIBANK.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "SBIN.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "AXISBANK.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "KOTAKBANK.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "MCX.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "BAJFINANCE.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "CHOLAFIN.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    "BANKBEES.NS": {"sector": "^NSEBANK", "group": "BFSI", "market": "INDIA_NSE"},
    # Auto & Industrials
    "M&M.NS": {"sector": "^CNXAUTO", "group": "AUTO_IND", "market": "INDIA_NSE"},
    "MARUTI.NS": {"sector": "^CNXAUTO", "group": "AUTO_IND", "market": "INDIA_NSE"},
    "BAJAJ-AUTO.NS": {"sector": "^CNXAUTO", "group": "AUTO_IND", "market": "INDIA_NSE"},
    "EICHERMOT.NS": {"sector": "^CNXAUTO", "group": "AUTO_IND", "market": "INDIA_NSE"},
    "LT.NS": {"sector": "^NSEI", "group": "CAP_GOODS", "market": "INDIA_NSE"},
    "HAL.NS": {"sector": "^NSEI", "group": "CAP_GOODS", "market": "INDIA_NSE"},
    "BEL.NS": {"sector": "^NSEI", "group": "CAP_GOODS", "market": "INDIA_NSE"},
    "ABB.NS": {"sector": "^NSEI", "group": "CAP_GOODS", "market": "INDIA_NSE"},
    "SIEMENS.NS": {"sector": "^NSEI", "group": "CAP_GOODS", "market": "INDIA_NSE"},
    # Metals & Energy
    "TATASTEEL.NS": {"sector": "^CNXMETAL", "group": "METALS_COMMODITY", "market": "INDIA_NSE"},
    "HINDALCO.NS": {"sector": "^CNXMETAL", "group": "METALS_COMMODITY", "market": "INDIA_NSE"},
    "SAIL.NS": {"sector": "^CNXMETAL", "group": "METALS_COMMODITY", "market": "INDIA_NSE"},
    "NATIONALUM.NS": {"sector": "^CNXMETAL", "group": "METALS_COMMODITY", "market": "INDIA_NSE"},
    "RELIANCE.NS": {"sector": "^CNXENERGY", "group": "ENERGY", "market": "INDIA_NSE"},
    "ONGC.NS": {"sector": "^CNXENERGY", "group": "ENERGY", "market": "INDIA_NSE"},
    "NTPC.NS": {"sector": "^CNXENERGY", "group": "ENERGY", "market": "INDIA_NSE"},
    "POWERGRID.NS": {"sector": "^CNXENERGY", "group": "ENERGY", "market": "INDIA_NSE"},
    "CHENNPETRO.NS": {"sector": "^CNXENERGY", "group": "ENERGY", "market": "INDIA_NSE"},
    # Consumer, Specialty & Broad ETFs
    "TITAN.NS": {"sector": "^NSEI", "group": "CONSUMER", "market": "INDIA_NSE"},
    "TRENT.NS": {"sector": "^NSEI", "group": "CONSUMER", "market": "INDIA_NSE"},
    "NYKAA.NS": {"sector": "^NSEI", "group": "CONSUMER", "market": "INDIA_NSE"},
    "ITC.NS": {"sector": "^CNXFMCG", "group": "FMCG_DEFENSIVE", "market": "INDIA_NSE"},
    "ADANIPORTS.NS": {"sector": "^NSEI", "group": "INFRA", "market": "INDIA_NSE"},
    "FINEORG.NS": {"sector": "^NSEI", "group": "CHEMICALS", "market": "INDIA_NSE"},
    "NIFTYBEES.NS": {"sector": "^NSEI", "group": "BROAD_ETF", "market": "INDIA_NSE"},
    "JUNIORBEES.NS": {"sector": "^NSEI", "group": "BROAD_ETF", "market": "INDIA_NSE"},
    "SMALLCAP.NS": {"sector": "^NSEI", "group": "BROAD_ETF", "market": "INDIA_NSE"},
    "ALPHA.NS": {"sector": "^NSEI", "group": "BROAD_ETF", "market": "INDIA_NSE"},
    # International & Global Macro ETFs / Proxies
    "MON100.NS": {"sector": "^IXIC", "group": "INTL_US_TECH", "market": "INTERNATIONAL"},
    "MAFANG.NS": {"sector": "^IXIC", "group": "INTL_US_TECH", "market": "INTERNATIONAL"},
    "MASPTOP50.NS": {"sector": "^GSPC", "group": "INTL_US_BROAD", "market": "INTERNATIONAL"},
    "GOLDBEES.NS": {"sector": "GC=F", "group": "INTL_GOLD_MACRO", "market": "INTERNATIONAL"},
}

NIFTY_250_RAW = [
    # --- Top 50 (Nifty 50 Benchmark) ---
    ("RELIANCE.NS", "Reliance Industries", "Energy & Retail"),
    ("TCS.NS", "Tata Consultancy Services", "IT Services"),
    ("HDFCBANK.NS", "HDFC Bank", "Banking"),
    ("ICICIBANK.NS", "ICICI Bank", "Banking"),
    ("INFY.NS", "Infosys Ltd", "IT Services"),
    ("BHARTIARTL.NS", "Bharti Airtel", "Telecom"),
    ("ITC.NS", "ITC Ltd", "FMCG"),
    ("SBIN.NS", "State Bank of India", "PSU Banking"),
    ("LT.NS", "Larsen & Toubro", "Infrastructure"),
    ("HINDUNILVR.NS", "Hindustan Unilever", "FMCG"),
    ("AXISBANK.NS", "Axis Bank", "Banking"),
    ("KOTAKBANK.NS", "Kotak Mahindra Bank", "Banking"),
    ("M&M.NS", "Mahindra & Mahindra", "Automobile"),
    ("MARUTI.NS", "Maruti Suzuki", "Automobile"),
    ("TMPV.NS", "Tata Motors (Passenger Vehicles)", "Automobile"),
    ("SUNPHARMA.NS", "Sun Pharma", "Pharma"),
    ("TITAN.NS", "Titan Company", "Consumer Discretionary"),
    ("BAJFINANCE.NS", "Bajaj Finance", "NBFC"),
    ("BAJAJFINSV.NS", "Bajaj Finserv", "Financial Services"),
    ("NTPC.NS", "NTPC Ltd", "Power / PSU"),
    ("POWERGRID.NS", "Power Grid Corp", "Power / PSU"),
    ("ONGC.NS", "ONGC", "Oil & Gas / PSU"),
    ("COALINDIA.NS", "Coal India", "Mining / PSU"),
    ("TATASTEEL.NS", "Tata Steel", "Metals"),
    ("JSWSTEEL.NS", "JSW Steel", "Metals"),
    ("HINDALCO.NS", "Hindalco Industries", "Metals"),
    ("ADANIENT.NS", "Adani Enterprises", "Conglomerate"),
    ("ADANIPORTS.NS", "Adani Ports & SEZ", "Infrastructure"),
    ("ULTRACEMCO.NS", "UltraTech Cement", "Cement"),
    ("GRASIM.NS", "Grasim Industries", "Materials"),
    ("WIPRO.NS", "Wipro", "IT Services"),
    ("HCLTECH.NS", "HCL Technologies", "IT Services"),
    ("TECHM.NS", "Tech Mahindra", "IT Services"),
    ("DRREDDY.NS", "Dr. Reddy's Labs", "Pharma"),
    ("CIPLA.NS", "Cipla Ltd", "Pharma"),
    ("APOLLOHOSP.NS", "Apollo Hospitals", "Healthcare"),
    ("ASIANPAINT.NS", "Asian Paints", "Paints / Consumer"),
    ("NESTLEIND.NS", "Nestle India", "FMCG"),
    ("BRITANNIA.NS", "Britannia Industries", "FMCG"),
    ("TATACONSUM.NS", "Tata Consumer Products", "FMCG"),
    ("EICHERMOT.NS", "Eicher Motors", "Automobile"),
    ("HEROMOTOCO.NS", "Hero MotoCorp", "Automobile"),
    ("BPCL.NS", "BPCL", "Oil & Gas / PSU"),
    ("BEL.NS", "Bharat Electronics", "Defense / PSU"),
    ("HAL.NS", "Hindustan Aeronautics", "Defense / PSU"),
    ("TRENT.NS", "Trent Ltd", "Retail"),
    ("VBL.NS", "Varun Beverages", "Beverages"),
    ("SHRIRAMFIN.NS", "Shriram Finance", "NBFC"),
    ("JSWENERGY.NS", "JSW Energy", "Power"),
    ("BAJAJ-AUTO.NS", "Bajaj Auto", "Automobile"),

    # --- Next 50 (Nifty Next 50 Large Caps) ---
    ("ABB.NS", "ABB India", "Capital Goods"),
    ("ADANIENSOL.NS", "Adani Energy Solutions", "Power"),
    ("ADANIGREEN.NS", "Adani Green Energy", "Renewable Energy"),
    ("ADANIPOWER.NS", "Adani Power", "Power"),
    ("AMBUJACEM.NS", "Ambuja Cements", "Cement"),
    ("BANKBARODA.NS", "Bank of Baroda", "PSU Banking"),
    ("BHEL.NS", "Bharat Heavy Electricals", "Capital Goods / PSU"),
    ("BOSCHLTD.NS", "Bosch Ltd", "Auto Ancillary"),
    ("CANBK.NS", "Canara Bank", "PSU Banking"),
    ("CHOLAFIN.NS", "Cholamandalam Inv", "NBFC"),
    ("COFORGE.NS", "Coforge Ltd", "IT Services"),
    ("COLPAL.NS", "Colgate-Palmolive India", "FMCG"),
    ("CUMMINSIND.NS", "Cummins India", "Capital Goods"),
    ("DABUR.NS", "Dabur India", "FMCG"),
    ("DIVISLAB.NS", "Divi's Laboratories", "Pharma"),
    ("DIXON.NS", "Dixon Technologies", "EMS / Electronics"),
    ("DLF.NS", "DLF Ltd", "Realty"),
    ("DMART.NS", "Avenue Supermarts (DMart)", "Retail"),
    ("GAIL.NS", "GAIL India", "Gas & Energy / PSU"),
    ("GODREJCP.NS", "Godrej Consumer Products", "FMCG"),
    ("HAVELLS.NS", "Havells India", "Consumer Electricals"),
    ("HDFCLIFE.NS", "HDFC Life Insurance", "Insurance"),
    ("ICICIGI.NS", "ICICI Lombard General Ins", "Insurance"),
    ("ICICIPRULI.NS", "ICICI Prudential Life Ins", "Insurance"),
    ("INDIANB.NS", "Indian Bank", "PSU Banking"),
    ("INDIGO.NS", "InterGlobe Aviation (IndiGo)", "Aviation"),
    ("INDUSINDBK.NS", "IndusInd Bank", "Banking"),
    ("IOC.NS", "Indian Oil Corp", "Oil & Gas / PSU"),
    ("IRFC.NS", "Indian Railway Finance", "Railway / PSU"),
    ("JINDALSTEL.NS", "Jindal Steel & Power", "Metals"),
    ("JIOFIN.NS", "Jio Financial Services", "Financial Services"),
    ("LTM.NS", "LTM Limited", "IT Services"),
    ("LUPIN.NS", "Lupin Ltd", "Pharma"),
    ("MARICO.NS", "Marico Ltd", "FMCG"),
    ("MAXHEALTH.NS", "Max Healthcare Institute", "Healthcare"),
    ("MOTHERSON.NS", "Samvardhana Motherson", "Auto Ancillary"),
    ("NAUKRI.NS", "Info Edge (Naukri)", "Platform / Tech"),
    ("NHPC.NS", "NHPC Ltd", "Power / PSU"),
    ("NMDC.NS", "NMDC Ltd", "Mining / PSU"),
    ("PFC.NS", "Power Finance Corp", "NBFC / PSU"),
    ("PIDILITIND.NS", "Pidilite Industries", "Chemicals"),
    ("PNB.NS", "Punjab National Bank", "PSU Banking"),
    ("POLYCAB.NS", "Polycab India", "Cables & Electricals"),
    ("RECLTD.NS", "REC Ltd", "NBFC / PSU"),
    ("SBILIFE.NS", "SBI Life Insurance", "Insurance"),
    ("SIEMENS.NS", "Siemens India", "Capital Goods"),
    ("TATAPOWER.NS", "Tata Power", "Power & Energy"),
    ("TORNTPHARM.NS", "Torrent Pharmaceuticals", "Pharma"),
    ("TVSMOTOR.NS", "TVS Motor Company", "Automobile"),
    ("VEDL.NS", "Vedanta Ltd", "Metals & Mining"),
    ("ETERNAL.NS", "Eternal Ltd (Zomato)", "Platform / Food Delivery"),

    # --- High-Conviction Midcaps 150 (Industrials, Defense, Tech, Chemicals, Financials) ---
    ("PERSISTENT.NS", "Persistent Systems", "IT Midcap"),
    ("KPITTECH.NS", "KPIT Technologies", "Automotive Software"),
    ("TATAELXSI.NS", "Tata Elxsi", "IT Design & Tech"),
    ("MPHASIS.NS", "Mphasis Ltd", "IT Services"),
    ("LTTS.NS", "L&T Technology Services", "ER&D IT"),
    ("CYIENT.NS", "Cyient Ltd", "Engineering Tech"),
    ("SONACOMS.NS", "Sona BLW Precision", "EV & Auto Ancillary"),
    ("MAZDOCK.NS", "Mazagon Dock Shipbuilders", "Defense / PSU"),
    ("COCHINSHIP.NS", "Cochin Shipyard", "Defense / PSU"),
    ("BDL.NS", "Bharat Dynamics", "Defense / PSU"),
    ("GRSE.NS", "Garden Reach Shipbuilders", "Defense / PSU"),
    ("ASTRAL.NS", "Astral Ltd", "Building Materials"),
    ("SUPREMEIND.NS", "Supreme Industries", "Plastic Products"),
    ("KEI.NS", "KEI Industries", "Cables & Wire"),
    ("RRKABEL.NS", "R R Kabel", "Cables & Electricals"),
    ("BHARATFORG.NS", "Bharat Forge", "Industrial Forgings"),
    ("SCHAEFFLER.NS", "Schaeffler India", "Bearings & Auto"),
    ("TIMKEN.NS", "Timken India", "Bearings"),
    ("SKFINDIA.NS", "SKF India", "Bearings"),
    ("AIAENG.NS", "AIA Engineering", "Capital Goods"),
    ("THERMAX.NS", "Thermax Ltd", "Clean Energy & Infra"),
    ("CGPOWER.NS", "CG Power & Industrial", "Electrical Goods"),
    ("KAYNES.NS", "Kaynes Technology", "EMS / Electronics"),
    ("SUZLON.NS", "Suzlon Energy", "Renewable Energy"),
    ("INOXWIND.NS", "Inox Wind", "Renewable Energy"),
    ("EXIDEIND.NS", "Exide Industries", "Batteries"),
    ("ARE&M.NS", "Amara Raja Energy & Mobility", "Batteries"),

    # --- Financials & Fintech Leaders ---
    ("MUTHOOTFIN.NS", "Muthoot Finance", "Gold Loans / NBFC"),
    ("MANAPPURAM.NS", "Manappuram Finance", "Gold Loans / NBFC"),
    ("M&MFIN.NS", "Mahindra Financial Services", "NBFC"),
    ("POONAWALLA.NS", "Poonawalla Fincorp", "NBFC"),
    ("SUNDARMFIN.NS", "Sundaram Finance", "NBFC"),
    ("LICHSGFIN.NS", "LIC Housing Finance", "Housing Finance"),
    ("CANFINHOME.NS", "Can Fin Homes", "Housing Finance"),
    ("AUBANK.NS", "AU Small Finance Bank", "Banking"),
    ("FEDERALBNK.NS", "Federal Bank", "Banking"),
    ("IDFCFIRSTB.NS", "IDFC First Bank", "Banking"),
    ("BANDHANBNK.NS", "Bandhan Bank", "Banking"),
    ("KARURVYSYA.NS", "Karur Vysya Bank", "Banking"),
    ("RBLBANK.NS", "RBL Bank", "Banking"),
    ("SBICARD.NS", "SBI Cards & Payment", "Financial Services"),
    ("ANGELONE.NS", "Angel One", "Broking / Fintech"),
    ("BSE.NS", "BSE Ltd", "Financial Exchange"),
    ("MCX.NS", "Multi Commodity Exchange", "Commodity Exchange"),
    ("CDSL.NS", "CDSL", "Depository Services"),
    ("CAMS.NS", "CAMS", "MF Registrar / Fintech"),
    ("HDFCAMC.NS", "HDFC Asset Management", "Asset Management"),
    ("NAM-INDIA.NS", "Nippon Life India AMC", "Asset Management"),
    ("POLICYBZR.NS", "PB Fintech (Policybazaar)", "Fintech / Insurtech"),
    ("PAYTM.NS", "One97 Communications (Paytm)", "Fintech"),
    ("NYKAA.NS", "FSN E-Commerce (Nykaa)", "Beauty / E-Commerce"),
    ("DELHIVERY.NS", "Delhivery Ltd", "Logistics"),

    # --- Pharma, Hospitals & Diagnostics ---
    ("ALKEM.NS", "Alkem Laboratories", "Pharma"),
    ("AUROPHARMA.NS", "Aurobindo Pharma", "Pharma"),
    ("BIOCON.NS", "Biocon Ltd", "Biopharma"),
    ("GLENMARK.NS", "Glenmark Pharmaceuticals", "Pharma"),
    ("IPCALAB.NS", "IPCA Laboratories", "Pharma"),
    ("LAURUSLABS.NS", "Laurus Labs", "Pharma & API"),
    ("MANKIND.NS", "Mankind Pharma", "Pharma"),
    ("NATCOPHARM.NS", "Natco Pharma", "Pharma"),
    ("SYNGENE.NS", "Syngene International", "Contract Research / CRO"),
    ("AJANTPHARM.NS", "Ajanta Pharma", "Pharma"),
    ("JBCHEPHARM.NS", "JB Chemicals & Pharma", "Pharma"),
    ("FORTIS.NS", "Fortis Healthcare", "Hospitals"),
    ("MEDANTA.NS", "Global Health (Medanta)", "Hospitals"),
    ("ASTERDM.NS", "Aster DM Healthcare", "Hospitals"),
    ("KIMS.NS", "Krishna Institute of Med Sci", "Hospitals"),
    ("METROPOLIS.NS", "Metropolis Healthcare", "Diagnostics"),
    ("LALPATHLAB.NS", "Dr. Lal PathLabs", "Diagnostics"),

    # --- Consumer Discretionary, Quick Service & Realty ---
    ("JUBLFOOD.NS", "Jubilant FoodWorks (Domino's)", "QSR"),
    ("DEVYANI.NS", "Devyani International (KFC)", "QSR"),
    ("WESTLIFE.NS", "Westlife Foodworld (McDonald's)", "QSR"),
    ("RADICO.NS", "Radico Khaitan", "Alcohol / Spirits"),
    ("UNITDSPR.NS", "United Spirits", "Alcohol / Spirits"),
    ("PAGEIND.NS", "Page Industries (Jockey)", "Apparel"),
    ("BATAINDIA.NS", "Bata India", "Footwear"),
    ("RELAXO.NS", "Relaxo Footwears", "Footwear"),
    ("VGUARD.NS", "V-Guard Industries", "Consumer Electricals"),
    ("CROMPTON.NS", "Crompton Greaves Consumer", "Consumer Electricals"),
    ("KAJARIACER.NS", "Kajaria Ceramics", "Ceramics"),
    ("CENTURYPLY.NS", "Century Plyboards", "Building Materials"),
    ("OBEROIRLTY.NS", "Oberoi Realty", "Realty"),
    ("GODREJPROP.NS", "Godrej Properties", "Realty"),
    ("PHOENIXLTD.NS", "The Phoenix Mills", "Malls & Realty"),
    ("BRIGADE.NS", "Brigade Enterprises", "Realty"),
    ("PRESTIGE.NS", "Prestige Estates Projects", "Realty"),
    ("SOBHA.NS", "Sobha Ltd", "Realty"),

    # --- Specialty Chemicals & Agro ---
    ("DEEPAKNTR.NS", "Deepak Nitrite", "Specialty Chemicals"),
    ("PIIND.NS", "PI Industries", "Agrochem / CSM"),
    ("SRF.NS", "SRF Ltd", "Chemicals & Packaging"),
    ("NAVINFLUOR.NS", "Navin Fluorine International", "Fluorochemicals"),
    ("TATACHEM.NS", "Tata Chemicals", "Inorganic Chemicals"),
    ("GUJENERGY.NS", "Gujarat Energy (Gujarat Gas)", "City Gas"),
    ("IGL.NS", "Indraprastha Gas", "City Gas"),
    ("MGL.NS", "Mahanagar Gas", "City Gas"),
    ("PETRONET.NS", "Petronet LNG", "Gas Transmission / PSU"),
    ("ATGL.NS", "Adani Total Gas", "City Gas"),
    ("COROMANDEL.NS", "Coromandel International", "Fertilizers & Nutrients"),
    ("CHAMBLFERT.NS", "Chambal Fertilisers", "Fertilizers"),
    ("UPL.NS", "UPL Ltd", "Crop Protection"),
    ("FACT.NS", "Fertilisers and Chem Travancore", "Fertilizers / PSU"),
    ("CLEAN.NS", "Clean Science & Technology", "Specialty Chemicals"),
    ("FINEORG.NS", "Fine Organic Industries", "Specialty Chemicals"),
    ("FLUOROCHEM.NS", "Gujarat Fluorochemicals", "Specialty Chemicals"),

    # --- Metals, Power, Energy & Shipping ---
    ("SAIL.NS", "Steel Authority of India", "Metals / PSU"),
    ("HINDZINC.NS", "Hindustan Zinc", "Metals / PSU"),
    ("NATIONALUM.NS", "National Aluminium (NALCO)", "Aluminium / PSU"),
    ("JINDALSAW.NS", "Jindal Saw", "Pipes & Tubes"),
    ("APLLTD.NS", "APL Apollo Tubes", "Structural Tubes"),
    ("RATNAMANI.NS", "Ratnamani Metals & Tubes", "Pipes & Tubes"),
    ("OIL.NS", "Oil India", "Oil & Gas Exploration / PSU"),
    ("MRPL.NS", "Mangalore Refinery & Petrochem", "Refinery / PSU"),
    ("CHENNPETRO.NS", "Chennai Petroleum Corp", "Refinery / PSU"),
    ("CESC.NS", "CESC Ltd", "Power Distribution"),
    ("TORNTPOWER.NS", "Torrent Power", "Power Generation"),
    ("SJVN.NS", "SJVN Ltd", "Hydro Power / PSU"),
    ("NLCINDIA.NS", "NLC India", "Lignite & Power / PSU"),

    # --- Auto Ancillary, Engineering & Logistics ---
    ("TIINDIA.NS", "Tube Investments of India", "Auto & Engineering"),
    ("BALKRISIND.NS", "Balkrishna Industries", "Tyres / Off-Highway"),
    ("MRF.NS", "MRF Ltd", "Tyres"),
    ("APOLLOTYRE.NS", "Apollo Tyres", "Tyres"),
    ("CEATLTD.NS", "CEAT Ltd", "Tyres"),
    ("ENDURANCE.NS", "Endurance Technologies", "Auto Components"),
    ("CONCOR.NS", "Container Corp of India", "Rail Logistics / PSU"),
    ("BLUESTARCO.NS", "Blue Star Ltd", "AC & Cooling"),
    ("VOLTAS.NS", "Voltas Ltd", "AC & Engineering"),
    ("ASHOKLEY.NS", "Ashok Leyland", "Commercial Vehicles"),
    ("CASTROLIND.NS", "Castrol India", "Lubricants"),
    ("UNIONBANK.NS", "Union Bank of India", "PSU Banking"),
    ("BSOFT.NS", "Birlasoft Ltd", "IT Services"),
    ("LATENTVIEW.NS", "Latent View Analytics", "Data & AI"),
    ("MAPMYINDIA.NS", "CE Info Systems (MapmyIndia)", "Geospatial Tech"),
    ("AFFLE.NS", "Affle India", "AdTech & Mobile"),
    ("ELGIEQUIP.NS", "Elgi Equipments", "Compressors & Machinery"),
    ("FINCABLES.NS", "Finolex Cables", "Cables & Wire"),
    ("CARBORUNIV.NS", "Carborundum Universal", "Abrasives & Ceramics"),
    ("RVNL.NS", "Rail Vikas Nigam", "Railways / PSU"),
    ("IRCON.NS", "Ircon International", "Railways / PSU"),
    ("HUDCO.NS", "Housing & Urban Dev Corp", "Housing Finance / PSU"),
    ("HINDPETRO.NS", "Hindustan Petroleum (HPCL)", "Oil & Gas / PSU"),
    ("KALYANKJIL.NS", "Kalyan Jewellers", "Jewellery & Retail"),
    ("TITAGARH.NS", "Titagarh Rail Systems", "Railways & Wagons"),
    ("TRITURBINE.NS", "Triveni Turbine", "Industrial Machinery"),
    ("PVRINOX.NS", "PVR INOX", "Entertainment & Media"),
    ("CREDITACC.NS", "CreditAccess Grameen", "Microfinance"),
    ("HOMEFIRST.NS", "Home First Finance", "Affordable Housing"),
    ("AAVAS.NS", "Aavas Financiers", "Housing Finance"),
    ("GLAXO.NS", "GlaxoSmithKline Pharma", "Pharma"),
    ("SANOFI.NS", "Sanofi India", "Pharma"),
]


def _classify_nifty250_category(cat: str) -> tuple[str, str]:
    c = cat.upper()
    if any(k in c for k in ("PHARMA", "HEALTH", "HOSPITAL", "DIAGNOSTIC", "CRO", "BIOPHARMA")):
        return "^CNXPHARMA", "PHARMA_DEFENSIVE"
    if any(k in c for k in ("IT ", "SOFTWARE", "TECH", "DATA", "AI", "ADTECH", "GEOSPATIAL")):
        return "^CNXIT", "IT_GLOBAL"
    if any(k in c for k in ("BANK", "NBFC", "FINANC", "INSUR", "EXCHANGE", "DEPOSITORY", "BROK", "ASSET", "HOUSING", "MICROFIN")):
        return "^NSEBANK", "BFSI"
    if any(k in c for k in ("AUTO", "TYRE", "COMMERCIAL VEHICLE", "EV ", "BEARING", "BATTER")):
        return "^CNXAUTO", "AUTO_IND"
    if any(k in c for k in ("METAL", "MINING", "ALUMIN", "PIPE", "TUBE", "STEEL")):
        return "^CNXMETAL", "METALS_COMMODITY"
    if any(k in c for k in ("POWER", "OIL", "GAS", "REFINERY", "ENERGY", "LIGNITE", "HYDRO", "RENEWABLE", "LUBRICANT")):
        return "^CNXENERGY", "ENERGY"
    if any(k in c for k in ("FMCG", "BEVERAGE", "ALCOHOL", "SPIRIT")):
        return "^CNXFMCG", "FMCG_DEFENSIVE"
    if any(k in c for k in ("DEFENSE", "CAPITAL GOODS", "ELECTRICAL", "CABLE", "FORGING", "COMPRESSOR", "TURBINE", "RAIL", "WAGON", "ENGINEERING", "EMS")):
        return "^NSEI", "CAP_GOODS"
    if any(k in c for k in ("CHEM", "FERTILIZER", "AGRO", "CROP", "FLUORO")):
        return "^NSEI", "CHEMICALS"
    if any(k in c for k in ("INFRA", "CEMENT", "LOGISTIC", "PORT", "AVIATION", "TELECOM", "CONGLOMERATE")):
        return "^NSEI", "INFRA"
    return "^NSEI", "CONSUMER"


def build_full_nifty250_universe_meta() -> dict:
    meta = {}
    for sym, name, cat in NIFTY_250_RAW:
        if sym in CORE_54_UNIVERSE_META:
            entry = dict(CORE_54_UNIVERSE_META[sym])
            entry["name"] = name
            entry["category"] = cat
            meta[sym] = entry
        else:
            sec_idx, grp = _classify_nifty250_category(cat)
            meta[sym] = {
                "sector": sec_idx,
                "group": grp,
                "market": "INDIA_NSE",
                "name": name,
                "category": cat,
            }
    for sym, info in CORE_54_UNIVERSE_META.items():
        if sym not in meta:
            entry = dict(info)
            entry.setdefault("name", sym.replace(".NS", ""))
            entry.setdefault("category", info["group"])
            meta[sym] = entry
    return meta


UNIVERSE_META = build_full_nifty250_universe_meta()
