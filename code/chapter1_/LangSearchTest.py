import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
url = "https://api.langsearch.com/v1/web-search"

payload = json.dumps({
  # "query": "今天广州天气",
  "query": "广州在晴朗天气下最值得去的旅游景点推荐及理由",
  "freshness": "noLimit",
  "summary": True,
  "count": 1
})
headers = {
  'Authorization': f'Bearer {os.getenv("LANGSEARCH_API_KEY")}',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

"""
{
    "code": 200,
    "log_id": "9c10d0be9f4f0d95",
    "msg": null,
    "data": {
        "_type": "SearchResponse",
        "queryContext": {
            "originalQuery": "广州在晴朗天气下最值得去的旅游景点推荐及理由"
        },
        "webPages": {
            "webSearchUrl": "https://langsearch.com/search?q=广州在晴朗天气下最值得去的旅游景点推荐及理由",
            "totalEstimatedMatches": null,
            "value": [{
                    "id": "https://api.langsearch.com/v1/#WebPages.1",
                    "name": "珠江边草坪露营地:直面广州塔&海心桥浪漫_腾讯新闻",
                    "url": "https://news.qq.com/rain/a/20260411A062CU00",
                    "displayUrl": "https://news.qq.com/rain/a/20260411A062CU00",
                    "snippet": "谁能拒绝在广州CBD坐拥一线珠江江景直面广州塔+海心桥的超大草坪露营呀✨不用跑远、不用花钱在二沙岛艺术公园把春天的江风、花海、落日、城市地标一次性拿捏春日限定美景✅\n超大绿油油草坪,肆意躺平野餐,氛",
                    "summary": "谁能拒绝在广州CBD坐拥一线珠江江景直面广州塔+海心桥的超大草坪露营呀✨不用跑远、不用花钱在二沙岛艺术公园把春天的江风、花海、落日、城市地标一次性拿捏春日限定美景✅\n超大绿油油草坪,肆意躺平野餐,氛围感直接拉满✅\n临江而坐,无遮挡看广州塔、海心桥,白天夜晚都巨出片✅\n春日花海盛放,蓝紫色飞燕草、三角梅开得超烂漫,随手拍都是油画感✅\n温柔江风拂面,暖阳洒在身上,彻底治愈打工人的疲惫详细地址:广州市越秀区二沙岛晴澜路72号(二沙岛艺术公园)交通攻略-\n地铁:APM线海心沙站A出口,步行10分钟过桥直达;5号线五羊邨站B口,骑行/步行20分钟到达-\n公交:二沙岛传祺公园站,下车步行300米即到-\n自驾/打车:导航定位艺术公园,岛内车位紧张,建议停海心沙停车场再步行️春日露营野餐必备清单▪️\n高颜值野餐垫/折叠椅,拍照巨出片▪️\n春日小零食、水果、饮品,拒绝明火,禁止烧烤▪️\n防晒帽、墨镜、防晒霜,江边紫外线超强▪️\n驱蚊液、垃圾袋,无痕露营爱护草坪▪️\n蓝牙音箱、泡泡机、风筝,增添春日氛围感狀避坑提醒:拒绝带汤汁、易融化、难清理的食物,吃完随手打包垃圾,保持草坪干净封神拍照机位\n1.\n草坪中央,低角度仰拍,和广州塔同框,春日松弛感拉满2.\n临江步道,以海心桥、珠江为背景,落日时分拍剪影绝了3.\n花海旁,让鲜花和江景、地标同框,春日氛围感直接拉满4.\n公园艺术雕塑旁,文艺又高级,轻松拍出ins风大片⏰最佳游玩时间15:00-19:00避开正午暴晒,傍晚邂逅绝美落日晚霞天黑后还能看广州塔夜景,一键解锁双重浪漫❌避坑小贴士1.\n周末/节假日人多,尽量早点去抢占江边C位草坪2.\n公园禁止明火、禁止大型固定帐篷、不可过夜3.\n岛内便利店少,美食、水、驱蚊用品提前备齐4.\n爱护花草草坪,垃圾全部自行带走,文明露营5.\n江边风大,带好薄外套,看好小朋友远离护栏不用远赴郊外",
                    "datePublished": "2026-04-11T09:57:44Z",
                    "dateLastCrawled": "2026-04-11T09:57:44Z"
                }
            ],
            "someResultsRemoved": true
        }
    }
}
"""