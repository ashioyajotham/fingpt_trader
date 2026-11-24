from asof.filing_statements import extract_financials


def test_extracts_stacked_million_bank_table_and_unscaled_eps():
    page1 = "Standard Chartered Bank Kenya Limited results for the year ended 31 December 2024."
    page3 = (
        " Financial performance summary\n"
        "                                                 31.12.2024             31.12.2023               2024 vs 2023\n"
        "                                                        KShs                   KShs             Better/(Worse)\n"
        "                                                      million                million                    %\n"
        " Total operating income                               50,677                  41,728                    21\n"
        " Profit before taxation                               28,208                  19,668                    43\n"
        " Profit after taxation                                20,060                  13,835                    45\n"
        " Earnings per share (KShs)                               52.7                    36.2                   46\n"
        " Loans and advances to customers                     151,647                163,162                     (7)\n"
        " Customer deposits                                   295,690                342,853                    (14)\n"
    )
    pages = [page1, "Dividend section", page3]
    fin = extract_financials(pages)
    assert fin["revenue"]["value"] == "50677000000"
    assert fin["revenue"]["prior_value"] == "41728000000"
    assert fin["revenue"]["period"] == "2024"
    assert fin["revenue"]["prior_period"] == "2023"
    assert fin["profit_before_tax"]["value"] == "28208000000"
    assert fin["profit_after_tax"]["value"] == "20060000000"
    # EPS must NOT be multiplied by 1,000,000.
    assert fin["eps"]["value"] == "52.7"
    assert fin["eps"]["prior_value"] == "36.2"
    assert fin["eps"]["unit"] == "KES/share"
    assert fin["loans_and_advances"]["value"] == "151647000000"
    assert fin["customer_deposits"]["value"] == "295690000000"
    for item in fin.values():
        for cite in item["citations"]:
            assert cite["quote"] in pages[cite["page"] - 1]


def test_extracts_thousands_corporate_table_and_negative_loss():
    page = (
        "Summary Consolidated Statement of Profit or Loss (KShs'000)\n"
        "                                    FY2025       FY2024\n"
        "Service revenue                  364,310,000  329,800,000\n"
        "Profit before tax                 (1,250,000)   84,700,000\n"
        "Total assets                     612,500,000  580,100,000\n"
        "Total equity                     215,400,000  201,000,000\n"
    )
    fin = extract_financials([page])
    assert fin["revenue"]["value"] == "364310000000"
    assert fin["profit_before_tax"]["value"] == "-1250000000"
    assert fin["total_assets"]["value"] == "612500000000"
    assert fin["total_equity"]["value"] == "215400000000"
    assert fin["revenue"]["period"] == "2025"
    assert fin["revenue"]["prior_period"] == "2024"


def test_real_nse_press_ad_quirks_two_column_cbk_and_prose_scale_guard():
    # Regression covering EABL "Kshs 'M" + "Net revenue", KPLC prose "KShs 14 billion" guard,
    # and CBK bank format ("Profit/ (loss) before tax and exceptional items", "EARNINGS PER SHARE - BASIC & DILUTED").
    eabl_like = (
        "Commentary: revenue reached Kshs 109.6bn despite macro headwinds.\n"
        "                        Kshs 'M        Kshs 'M          Management commentary on liquidity.\n"
        "                        2023           2022\n"
        "Net revenue             109,649        109,409          Strong brand momentum across East Africa.\n"
        "Profit for the year      12,323         15,574          Resilient margins.\n"
    )
    eabl = extract_financials([eabl_like])
    assert eabl["revenue"]["value"] == "109649000000"
    assert eabl["profit_after_tax"]["value"] == "12323000000"

    cbk_like = (
        "Audited financial statements for the year ended 31 December 2021\n"
        "KShs. '000\n"
        "Shareholders' funds rose steadily since 2018 due to retained earnings.\n"
        "TOTAL SHAREHOLDERS' EQUITY                         52,479,395     50,218,630\n"
        "Profit/ (loss) before tax and exceptional items    12,141,504      7,018,311\n"
        "Profit/ (loss) after tax and exceptional items      8,980,536      5,329,501\n"
        "EARNINGS PER SHARE - BASIC & DILUTED (KShs)             23.32          13.66\n"
    )
    cbk = extract_financials([cbk_like])
    assert cbk["total_equity"]["value"] == "52479395000"
    assert cbk["profit_before_tax"]["value"] == "12141504000"
    assert cbk["profit_after_tax"]["value"] == "8980536000"
    assert cbk["eps"]["value"] == "23.32"
