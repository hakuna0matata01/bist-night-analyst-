# BIST Akşam Taraması — Faz 1 (v4)

## v4 — Sağlamlık, Dürüst Backtest ve Test Altyapısı

**Azami puan nedir?** Pozitif kriter sayısı **7**'dir; iki risk cezası yalnızca puanı düşürür, hiçbir
zaman artırmaz. Dolayısıyla azami puan **7**'dir. **6** değeri yalnızca BIST100 endeksi alınamadığında
görülür (Göreli Güç kriteri hesaplanamaz) ve arayüz bunu açıkça uyarır.

| # | Zayıflık | Çözüm |
|---|----------|-------|
| 1 | Hisse indeksi tz'li, endeks tz'siz gelince `reindex` hata verip **tüm hisseleri sessizce düşürebiliyordu** | `normalize_index()` ile tüm seriler tz'siz günlüğe çevrilir |
| 2 | Tek hisse indirmede (ör. "Kendi listem"de tek kod) yfinance'in çok seviyeli sütunu `KeyError` ile hisseyi kaybettiriyordu | `extract_ticker_frame()` her iki sütun biçimini de işler |
| 3 | Geçici Yahoo hatası **2 saat boyunca önbelleğe** alınıyordu; hata nedeni gizleniyordu | Grup başına 3 deneme + eksikler için ikinci geçiş; tam başarısızlık önbelleğe alınmaz, neden listelenir |
| 4 | Seans bitmeden çalıştırılınca **yarım mum** hesaba giriyordu | 18:15 TSİ öncesi bugünkü mum otomatik hariç tutulur, kullanıcı bilgilendirilir |
| 5 | İşlem görmeyen/askıdaki hisse eski veriyle puanlanıyordu | Son veri evrenin gerisindeyse ya da son gün hacim 0 ise "analiz edilemedi" |
| 6 | Eski bir bedelsiz izi hisseyi tümden dışarıda bırakıyordu | Son sıçramadan sonrası yeterliyse geçmiş kırpılıp analiz sürer (liste halinde gösterilir) |
| 7 | ADX yalnızca **güç** ölçer, yön değil; düşüş trendinde de puan verebilirdi | ADX puanı için `+DI > −DI` şartı eklendi (`ADX_REQUIRE_DI_DIRECTION`) |
| 8 | Yönsüz Bollinger sıkışması tek başına "Seviye" puanı veriyordu | Sıkışma yalnızca fiyat EMA20 üzerindeyken sayılır (`SQUEEZE_REQUIRES_BULLISH`) |
| 9 | RSI ısınma dönemi 50'ye doldurulup sahte değerler üretiyordu | Isınma NaN kalır; backtest ilk 60 barı atlar |
| 10 | Backtest "sinyal kapanışından ertesi kapanışa" ölçüyordu — **fiilen yapılamayan** bir işlem | Ek **tetik bazlı** senaryo: ertesi gün tetik görülürse tetik/açılıştan alım, kapanışta satım; tavan açılış hariç; işlem maliyeti düşülür |
| 11 | Başarı oranı **baz oran olmadan** yorumlanıyordu | Tüm hisse-günlerinden baz oran ve **fark (lift)** gösterilir |
| 12 | Aynı gün çıkan sinyaller bağımsız değil; Wilson aralığı fazla dar kalıyordu | Tüm-evren için **gün-kümelenmeli** güven aralığı |
| 13 | Puanın gerçekten ayırt edici olup olmadığı görülemiyordu | "Puan basamağına göre sonuçlar" tablosu (yüksek puan gerçekten daha mı iyi?) |
| 14 | Min. puan/likidite gibi filtreler değişince **yeniden tarama** gerekiyordu | Görüntüleme filtreleri anında uygulanır; tarihsel özetler havuzdan türetilir |
| 15 | Eşit puanda sıralama gürültülü "tarihsel başarı"ya dayanıyordu | Seçilebilir ölçüt (göreli güç, ciro, hacim oranı, tetiğe yakınlık) |
| 16 | `requirements.txt` (`streamlit>=1.38`) kodun kullandığı `width="stretch"` ile uyumsuzdu | `streamlit>=1.50`, `pandas>=2.1` |
| 17 | Grafikte MACD ve ±DI yoktu, hafta sonu boşlukları vardı | MACD ve ±DI panelleri, hafta sonu boşlukları kaldırıldı |
| 18 | Mantık arayüze gömülüydü, test yoktu | `engine.py` ayrıldı; `tests/` altında 22 test (nedensellik/look-ahead, azami puan, tetik matematiği, veri hijyeni) |

> **Kural değişikliği uyarısı:** 7 ve 8 numaralı maddeler puanlama kurallarını bilinçli olarak biraz
> daha seçici yapar. v3 davranışına dönmek için `engine.py` başındaki iki anahtarı `False` yapın.
>
> **Hâlâ geçerli sınırlar:** hisse listesi bugünün hisseleridir (hayatta kalma yanlılığı sonuçları
> iyimser gösterebilir); Yahoo verisi gecikmeli/gayri resmi olabilir; tetik backtest'i emir derinliğini
> ve tavan/taban kilitlerini tam modellemez.

## Testler

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Proje Yapısı

- `app.py` — Streamlit arayüzü, veri indirme (yeniden deneme/önbellek), sunum
- `engine.py` — göstergeler, puanlama, veri hijyeni, backtest (Streamlit'ten bağımsız)
- `tests/test_engine.py` — motor testleri
- `data/bist_tickers.txt` — tarama evreni

Piyasa kapandıktan sonra (örn. saat 20:00) çalıştırılan, BIST hisselerini
teknik kriterlere göre puanlayıp ertesi gün için kısa bir **izleme listesi**
(watchlist) çıkaran bir Streamlit uygulaması.

> ⚠️ Bu araç yatırım tavsiyesi değildir. Sadece teknik kriterlere dayalı bir
> ön-eleme/tarama aracıdır.

## Puanlama Mantığı (v4 — azami 7 puan, risk cezalı; endeks verisi yoksa azami 6)

**Pozitif kriterler (her biri +1):**

| # | Kriter | Açıklama |
|---|--------|----------|
| 1 | Trend | Fiyat > EMA20 > EMA50 (boğa dizilişi — sadece fiyat>EMA değil) |
| 2 | Momentum | RSI(14) 40-65 bandında ve yükseliyor **veya** (Seviye kriteri sağlanmışsa) RSI 65-80 arası ve yükseliyor |
| 3 | MACD | MACD çizgisi sinyalin üzerinde, histogram güçleniyor |
| 4 | Hacim | Günlük hacim 20g ortalamanın **1.3x** üzerinde + fiyat yukarı |
| 5 | Seviye | **Günlük en yüksek (High) fiyatlara dayalı** direnç kırılımı/yakınlığı ya da **yükseliş yanlı** sıkışma (düşük volatilite + fiyat > EMA20) |
| 6 | ADX | ADX(14) ≥ eşik **ve +DI > −DI** (güç + yön) — klasik Wilder formülüyle hesaplanır |
| 7 | Göreli Güç | Son 20 günde **BIST100 endeksinden** daha güçlü performans (endeks verisi alınamazsa bu kriter hesaplanamaz, azami puan 6'ya düşer) |

**Risk cezaları (her biri -1, toplam puan 0'ın altına inmez):**

| Bayrak | Açıklama |
|--------|----------|
| Aşırı uzama | Fiyat, EMA20'nin %10+ üzerinde (kovalama/geri çekilme riski) |
| Marj sınırı | Bugünkü değişim zaten ±%9+ (yarın için gap/tepki riski) |

**Ek bağlam ve filtreler:**
- **Piyasa rejimi:** BIST100 endeksinin kendi trendi (EMA50 üzerinde mi) üstte
  bir bilgi/uyarı panosu olarak gösterilir. Endeks zayıfken bireysel sinyaller
  istatistiksel olarak daha az güvenilir olma eğilimindedir.
- **Likidite filtresi:** 20 günlük ortalama TL cirosu, sol menüden
  belirlenen eşiğin altındaki hisseler izleme listesinden (isteğe bağlı,
  varsayılan açık) hariç tutulur — ince/az işlem gören hisselerde teknik
  sinyaller güvenilmezdir ve pratikte alım/satımı da zor olabilir.
- **Tarihsel doğrulama:** Tüm taranan hisselerin geçmişi (ısınma sonrası) tek havuzda toplanır.
  Her sinyal için iki ölçüt verilir: (a) sinyal kapanışından ertesi kapanışa **brüt** getiri/pozitif oranı,
  (b) **tetik bazlı** gerçekçi senaryo (ertesi gün tetik görülürse alım, kapanışta satım, tur başına
  2× işlem maliyeti düşülür). Sonuçlar **baz oranla** kıyaslanır; aralıklar tüm evrende gün-kümelenmeli,
  tek hissede Wilson aralığıdır. **Bu geçmiş performanstır, gelecekteki sonucu garanti etmez.**

## Giriş Tetik Seviyesi

Orijinal Faz 1 planındaki *"her biri için giriş tetik seviyesi (dünkü direncin
X kuruş üzeri) not et"* adımı, sonuç tablosunda ayrı sütunlar olarak hesaplanır:

- **Referans Direnç:** O gün için izlenecek direnç — hisse bugün henüz
  direncini kırmadıysa, dünden bugüne taşınan 20 günlük direnç seviyesi; eğer
  bugün zaten kırılım yaptıysa (daha önceki direncin üzerinde kapandıysa),
  bugünün kapanışı yeni referans olur (devamlılık teyidi aranır).
- **Giriş Tetik Fiyatı:** Referans Direnç × (1 + tampon%). Yarın fiyat bu
  seviyeyi yukarı keserse (ve Faz 2'deki RVOL/yön teyidi de sağlanırsa) giriş
  düşünülebilecek fiyat budur.
- **Tetiğe Uzaklık %:** Bugünkü kapanışın tetik fiyatına olan yüzde mesafesi.

**Neden sabit kuruş değil de yüzde tampon?** BIST'te hisseler birkaç TL'den
binlerce TL'ye kadar çok farklı fiyat seviyelerinde işlem görüyor; 634 hissenin
tamamında anlamlı kalması için tampon yüzde olarak uygulanır (sol menüden
ayarlanabilir, varsayılan %0.5). Bu seviye grafik üzerinde turuncu kesikli
çizgi olarak da işaretlenir ve İzleme Listesi CSV'sine dahildir — ertesi sabah
Faz 2'de doğrudan kullanılabilir.

Bu, stratejinin **Faz 2** (sabah 10:30 sonrası Relative Volume teyidi ve
alım) adımından önceki hazırlık aşamasıdır; Faz 2 ayrı bir uygulama/modül
olarak eklenebilir.

## Yerelde Çalıştırma

```bash
git clone <repo-url>
cd bist-night-analyst
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Tarayıcıda otomatik olarak `http://localhost:8501` açılır.

## GitHub'a Yükleme

```bash
cd bist-screener
git init
git add .
git commit -m "BIST akşam taraması - Faz 1"
git branch -M main
git remote add origin https://github.com/KULLANICI_ADIN/bist-screener.git
git push -u origin main
```

## Streamlit Community Cloud'a Dağıtım

1. [share.streamlit.io](https://share.streamlit.io) adresine GitHub hesabınızla giriş yapın.
2. **"New app"** butonuna tıklayın.
3. Az önce push ettiğiniz repoyu, branch'i (`main`) ve ana dosyayı (`app.py`) seçin.
4. **Deploy** butonuna basın. Birkaç dakika içinde uygulamanız canlıda olur
   (örn. `https://bist-screener-xxxx.streamlit.app`).
5. Her `git push` sonrası uygulama otomatik olarak güncellenir.

Her akşam saat 20:00 civarı bu linke girip "Taramayı Başlat"a basmanız yeterli.

## Hisse Listesi

Uygulama üç kaynaktan hisse listesi alabilir (sol menüden seçilir):

- **Tüm BIST Hisseleri** *(varsayılan)* — `data/bist_tickers.txt` dosyasından
  okunur. Bu dosya, yüklenen "BIST-tumhisse.xlsx" (isyatirim) listesinden
  çıkarılan **634 hisse kodunu** içerir, bir satırda bir kod olacak şekilde.
  Listeyi güncellemek için bu `.txt` dosyasını düzenlemeniz yeterli — kod
  değişikliği gerekmez.
- **Örnek alt küme (BIST30)** — hızlı test/deneme için küçük, sabit bir liste.
- **Kendi listem** — virgülle ayrılmış, istediğiniz hisseler (`.IS` uzantısı
  otomatik eklenir).

### Tam liste taramasında performans

634 hissenin tamamını taramak, 30 hisseyi taramaktan çok daha uzun sürer:

- İndirme, Yahoo Finance hız sınırlarını aşmamak için **100'lük gruplar**
  halinde ve paralel (`threads=True`) yapılır (`DOWNLOAD_CHUNK_SIZE` ile
  `app.py` içinden ayarlanabilir; her grup 3 kez denenir).
- Sonuçlar **2 saat** boyunca önbellekte tutulur (`fetch_batch` üzerindeki
  `ttl=7200`); aynı akşam tekrar tekrar basmanıza gerek kalmaz.
- Likit olmayan/az işlem gören bazı kodlarda veri gelmeyebilir; bunlar
  "Analiz edilemeyen hisseler" kısmında listelenir, uygulamayı durdurmaz.
- Daha hızlı yinelemeli test için "Örnek alt küme (BIST30)" seçeneğini
  kullanmanızı, tam taramayı günde bir kez (akşam) çalıştırmanızı öneririz.

## Özelleştirme

- **Arayüzden ayarlanabilenler:** yeniden tarama gerektirenler — hisse listesi, veri aralığı, ADX eşiği,
  tetik tamponu; **anında uygulananlar** — minimum puan, likidite eşiği/hariç tutma, eşit puan sıralaması,
  işlem maliyeti.
- **Kodda sabit olan eşikler** (`engine.py` başındaki sabitler): RSI bandı
  (40-65), kırılım durumunda izin verilen üst RSI sınırı (`RSI_BREAKOUT_UPPER`,
  80), hacim çarpanı (1.3x), direnç toleransı (%2), aşırı uzama eşiği
  (`OVEREXTENSION_PCT`, %10), marj sınırı eşiği (`NEAR_LIMIT_PCT`, %9),
  göreli güç bakış penceresi (20 gün), minimum backtest sinyal sayısı
  (`MIN_BACKTEST_SIGNALS`, 5), minimum tüm-evren backtest sinyal sayısı
  (`MIN_AGGREGATE_BACKTEST_SIGNALS`, 30), şüpheli fiyat sıçraması eşiği
  (`SUSPICIOUS_JUMP_PCT`, %30). Kendi stratejinize göre bu sabitleri
  değiştirebilirsiniz.
- **Endeks referansı:** Göreli güç ve piyasa rejimi notu `INDEX_TICKER =
  "XU100.IS"` (BIST100) üzerinden hesaplanır; isterseniz başka bir endekse
  çevirebilirsiniz.
- **Hisse listesini güncelleme:** isyatirim.com veya BIST'in resmi
  kaynaklarından yeni bir liste alıp `data/bist_tickers.txt` dosyasının
  içeriğini (bir satır = bir hisse kodu, `.IS` olmadan) değiştirerek
  güncelleyebilirsiniz.

## Veri Kaynağı

Veriler [Yahoo Finance](https://finance.yahoo.com) üzerinden `yfinance`
kütüphanesi ile çekilir (BIST hisseleri `.IS` uzantısıyla, örn. `THYAO.IS`).
Yahoo Finance verileri gecikmeli/gayri resmi olabilir; kritik kararlar için
resmi veri kaynaklarıyla teyit edin.

## Olası Sonraki Adımlar

- Faz 2 (sabah RVOL teyidi + VWAP kontrolü) için ayrı bir sayfa/sekme eklemek
- Tarama sonuçlarını otomatik olarak her akşam GitHub Actions ile çalıştırıp
  sonucu bir CSV/Telegram bildirimi olarak almak
- Walk-forward (zaman içinde eşik optimizasyonu olmadan, yalnızca dönemsel kararlılık) doğrulama
- Hayatta kalma yanlılığını gidermek için tarihsel hisse listeleri
- Sektör bazlı gruplama ve sektör göreli gücü (tek tek hisse yerine sektör
  rotasyonunu da dikkate almak)
