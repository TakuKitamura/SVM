#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
気象庁から過去の気象データを CSV 形式でダウンロードする。
API が提供されていないので、ウェブページを参考にスクリプトを作成した。

http://www.data.jma.go.jp/gmd/risk/obsdl/index.php

とりあえず時別値のダウンロードのみ対応。
"""

from datetime import date
import urllib.request
import lxml.html


def encode_data(data):
    return urllib.parse.urlencode(data).encode(encoding='ascii')

def get_phpsessid():
    URL="http://www.data.jma.go.jp/gmd/risk/obsdl/index.php"
    xml = urllib.request.urlopen(URL).read().decode("utf-8")
    tree = lxml.html.fromstring(xml)
    return tree.cssselect("input#sid")[0].value


# 観測地点選択
def get_station(pd=0):
    assert type(pd) is int and pd >= 0

    URL="http://www.data.jma.go.jp/gmd/risk/obsdl/top/station"
    data = encode_data({"pd": "%02d" % pd})
    xml = urllib.request.urlopen(URL, data=data).read().decode("utf-8")
    tree = lxml.html.fromstring(xml)

    def kansoku_items(bits):
        return dict(rain=(bits[0] == "1"),
                    wind=(bits[1] == "1"),
                    temp=(bits[2] == "1"),
                    sun =(bits[3] == "1"),
                    snow=(bits[4] == "1"))

    def parse_station(dom):
        stitle = dom.get("title").replace("：", ":")
        title = dict(filter(lambda y: len(y) == 2,
                            map(lambda x: x.split(":"), stitle.split("\n"))))

        name    = title["地点名"]
        stid    = dom.cssselect("input[name=stid]")[0].value
        stname  = dom.cssselect("input[name=stname]")[0].value
        kansoku = kansoku_items(dom.cssselect("input[name=kansoku]")[0].value)
        assert name == stname
        return (stname, dict(id=stid, flags=kansoku))

    def parse_prefs(dom):
        name = dom.text
        prid = int(dom.cssselect("input[name=prid]")[0].value)
        return (name, prid)

    if pd > 0:
        stations = dict(map(parse_station, tree.cssselect("div.station")))
    else:
        stations = dict(map(parse_prefs, tree.cssselect("div.prefecture")))
    return stations


# 観測項目選択
def get_aggrgPeriods():
    URL="http://www.data.jma.go.jp/gmd/risk/obsdl/top/element"
    xml = urllib.request.urlopen(URL).read().decode("utf-8")  # HTTP GET
    tree = lxml.html.fromstring(xml)

    def parse_periods(dom):
        if dom.find("label") is not None:
            val = dom.find("label/input").attrib["value"]
            key = dom.find("label/span").text
            rng = None
        else:
            val = dom.find("input").attrib["value"]
            key = dom.find("span/label").text
            rng = list(map(lambda x: int(x.get("value")),
                           dom.find("span/select").getchildren()))
        return (key, (val, rng))

    perdoms = tree.cssselect("#aggrgPeriod")[0].find("div/div").getchildren()
    periods = dict(map(parse_periods, perdoms))
    return periods

def get_elements(aggrgPeriods=9, isTypeNumber=1):
    URL="http://www.data.jma.go.jp/gmd/risk/obsdl/top/element"
    data = encode_data({"aggrgPeriod": aggrgPeriods,
                        "isTypeNumber": isTypeNumber})
    xml = urllib.request.urlopen(URL, data=data).read().decode("utf-8")
    open("tmp.html", "w").write(xml)
    tree = lxml.html.fromstring(xml)

    boxes = tree.cssselect("input[type=checkbox]")
    options, items = boxes[0:4], boxes[4:]

    def parse_items(dom):
        if "disabled" in dom.attrib: return None
        if dom.name == "kijiFlag": return None
        name     = dom.attrib["id"]
        value    = dom.attrib["value"]
        options  = None
        select = dom.getnext().find("select")
        if select is not None:
            options = list(map(lambda x: int(x.get("value")),
                               select.getchildren()))
        return (name, (value, options))

    items = dict(filter(lambda x: x, map(parse_items, items)))
    return items


def download_hourly_csv(phpsessid, station, element, begin_date, end_date):
    params = {
        "PHPSESSID": phpsessid,
        # 共通フラグ
        "rmkFlag": 1,        # 利用上注意が必要なデータを格納する
        "disconnectFlag": 1, # 観測環境の変化にかかわらずデータを格納する
        "csvFlag": 1,        # すべて数値で格納する
        "ymdLiteral": 1,     # 日付は日付リテラルで格納する
        "youbiFlag": 0,      # 日付に曜日を表示する
        "kijiFlag": 0,       # 最高・最低（最大・最小）値の発生時刻を表示
        # 時別値データ選択
        "aggrgPeriod": 9,    # 時別値
        "stationNumList": '["%s"]' % station,      # 観測地点IDのリスト
        "elementNumList": '[["%s",""]]' % element, # 項目IDのリスト
        "ymdList": '["%d", "%d", "%d", "%d", "%d", "%d"]' % (
            begin_date.year,  end_date.year,
            begin_date.month, end_date.month,
            begin_date.day,   end_date.day),       # 取得する期間
        "jikantaiFlag": 0,        # 特定の時間帯のみ表示する
        "jikantaiList": '[1,24]', # デフォルトは全部
        "interAnnualFlag": 1,     # 連続した期間で表示する
        # 以下、意味の分からないフラグ類
        "optionNumList": [],
        "downloadFlag": "true",   # CSV としてダウンロードする？
        "huukouFlag": 0,
    }

    URL="http://www.data.jma.go.jp/gmd/risk/obsdl/show/table"
    data = encode_data(params)
    csv = urllib.request.urlopen(URL, data=data).read().decode("shift-jis")
    print(csv)


if __name__ == "__main__":
    # print(get_aggrgPeriods())
    element = get_elements(get_aggrgPeriods()["時別値"][0])["気温"][0]
    station = get_station(get_station(0)["東京"])["東京"]["id"]
    phpsessid = get_phpsessid()

    download_hourly_csv(phpsessid, station, element,
                        date(2014, 1, 1), date(2014, 1, 31))
