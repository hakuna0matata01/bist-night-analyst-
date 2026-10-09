# BIST Akşam Taraması — Faz 1

Piyasa kapandıktan sonra (örn. saat 20:00) çalıştırılan, BIST hisselerini
teknik kriterlere göre puanlayıp ertesi gün için kısa bir **izleme listesi**
(watchlist) çıkaran bir Streamlit uygulaması.

> ⚠️ Bu araç yatırım tavsiyesi değildir. Sadece teknik kriterlere dayalı bir
> ön-eleme/tarama aracıdır.

## Puanlama Mantığı (5 üzerinden)

| # | Kriter | Açıklama |
|---|--------|----------|
| 1 | Trend | Fiyat > EMA20 ve Fiyat > EMA50 |
| 2 | Momentum | RSI(14) 40-65 bandında ve yükseliyor |
| 3 | MACD | MACD çizgisi sinyalin üzerinde, histogram güçleniyor |
| 4 | Hacim | Günlük hacim 20 günlük ortalamanın üzerinde + fiyat yukarı |
| 5 | Seviye | Direnç kırılımı/yakınlığı ya da sıkışma (düşük volatilite) sonrası |

Bu, stratejinin **Faz 2** (sabah 10:30 sonrası Relative Volume teyidi ve
alım) adımından önceki hazırlık aşamasıdır; Faz 2 ayrı bir uygulama/modül
olarak eklenebilir.

## Yerelde Çalıştırma

```bash
git clone <repo-url>
cd bist-screener
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
  `app.py` içinden ayarlanabilir).
- Sonuçlar **2 saat** boyunca önbellekte tutulur (`fetch_batch` üzerindeki
  `ttl=7200`); aynı akşam tekrar tekrar basmanıza gerek kalmaz.
- Likit olmayan/az işlem gören bazı kodlarda veri gelmeyebilir; bunlar
  "Analiz edilemeyen hisseler" kısmında listelenir, uygulamayı durdurmaz.
- Daha hızlı yinelemeli test için "Örnek alt küme (BIST30)" seçeneğini
  kullanmanızı, tam taramayı günde bir kez (akşam) çalıştırmanızı öneririz.

## Özelleştirme

- **Eşik değerleri:** `analyze_stock()` fonksiyonu içindeki RSI bandı (40-65),
  hacim çarpanı, direnç toleransı (%2) gibi değerler koda gömülüdür; kendi
  stratejinize göre bu sabitleri değiştirebilirsiniz.
- **Minimum puan filtresi:** Arayüzden (sidebar) canlı olarak ayarlanabilir.
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
- Backtest modülü ekleyerek puanlama eşiklerinin geçmiş performansını ölçmek
