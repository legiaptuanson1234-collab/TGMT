# trafficbot.py - Chatbot Giao Thông (rule-based tiếng Việt, chạy 100% local)
# Hỏi bất cứ lúc nào trong khi video đang chạy -> tra TrafficStore -> trả lời thống kê real-time.
import re
from datetime import datetime

from livestore import CLASS_NAMES, ALL_CLASSES


def _fmt_range(sec_start, sec_end):
    """'08:15 - 08:30' dạng phút video, hoặc 'từ 0 đến 90s'."""
    m0, s0 = divmod(int(sec_start), 60)
    m1, s1 = divmod(int(sec_end) if sec_end else 0, 60)
    return "%02d:%02d - %02d:%02d (ph)" % (m0, s0, m1, s1)


def _counts_md(counts, total):
    if total <= 0:
        return "chưa có xe nào đi qua"
    parts = []
    order = [1, 0, 2, 3, 4]  # xe máy thường nhiều nhất -> hiện đầu
    for c in order:
        if counts.get(c):
            pct = round(100.0 * counts[c] / total, 1)
            parts.append("**%s: %d** (%.0f%%)" % (CLASS_NAMES[c], counts[c], pct))
    return ", ".join(parts)


def _status_line(store):
    st = store.stats()
    return ("> 🚦 Trạng thái: **%s** | Xuôi: %d xe | Hiện trong khung: **%d xe** | %d/phút (5ph gần nhất)"
            % (st["status"], st["total"], st["active_now"], st["per_minute"]))


def ask(store, message, video_sec_now):
    """Trả lời câu hỏi. store: TrafficStore đang chạy. video_sec_now: vị trí hiện tại (giây)."""
    m = re.sub(r"\s+", " ", (message or "").strip().lower())
    if not m:
        return help_reply(store)
    st = store.stats()

    # ---------- không có dữ liệu ----------
    if not st["events"] and not st["running"]:
        return ("Chưa có dữ liệu nào. Hãy **nạp video → vẽ vạch/vùng → bấm CHẠY AI** "
                "rồi hỏi tôi — ví dụ: *'nay có bao nhiêu xe?'*, *'từ 1h sang có bao nhiêu xe máy?*' "
                "Hay dùng **/giúp đỡ** để xem lệnh.\n" + _status_line(store))

    # ---------- lệnh hệ thống ----------
    if m in ("giúp", "giúp đỡ", "help", "/giúp đỡ", "?", "hỏi gì được") or "giúp" in m:
        return help_reply(store)

    if "csv" in m or "báo cáo" in m or "baocao" in m or "xuất" in m:
        path = store.write_summary_csv()
        csv = st.get("detail_csv")
        msg = ("✅ Đã xuất thống kê theo phút: `%s`\n"
               "📡 File sự kiện **ghi liên tục** (real-time): `%s` — mở Excel ngay được, "
               "không phải chờ kết thúc." % (path, csv or "(chưa chạy)"))
        return msg

    if "đặt tên" in m or "đổi tên" in m or "báo cáo tên" in m:
        return "Tên báo cáo được đặt ở Tab 5 (Xuất báo cáo) — tôi chỉ thống kê dữ liệu. " + _status_line(store)

    # ---------- 'nay hôm nay hôm kia bao nhiêu xe' ----------
    today = "nay" in m or "hôm nay" in m or "homenay" in m
    # loại trừ: câu có phạm vi thời gian HOẶC hỏi về 1 LOẠI xe cụ thể (xử lý bên dưới)
    cls_word = any(k in m for k in ("xe máy", "xe may", "xe tải", "xe tai", "xe bus",
                                    "xe ba gác", "xe ba giac", "ô tô", "o to"))
    has_range = bool(re.search(r"\d+\s*(?:phút|phut|ph)\b.*?\d+\s*(?:phút|phut|ph)?", m) or re.search(r"\d+[:\-]\d+", m))
    if today or ("bao nhiêu xe" in m and not has_range and not cls_word):
        q = store.query_range(0, video_sec_now)
        top = sorted(((c, q["counts"].get(c, 0)) for c in ALL_CLASSES), key=lambda x: -x[1])
        top = [t for t in top if t[1] > 0]
        peak = st["peak"]
        peak_md = ""
        if peak:
            pm, ps = divmod(peak["minute"] * 60, 60)
            peak_md = ("\n📊 **Đỉnh cao nhất**: phút %d–%d (%d xe/phút)."
                       % (peak["minute"], peak["minute"] + 1, peak["total"]))
        reply = ("📈 **Tổng từ khi bắt đầu đến giờ** (%s): **%d xe**\n%s%s\n\n%s"
                 % (_fmt_range(0, int(q["end"])), q["total"], _counts_md(q["counts"], q["total"]),
                    peak_md, _status_line(store)))
        if q["avg_speed"] > 0:
            reply += "\n⚡ Tốc độ trung bình: **%.1f km/h** (thang pixel→mét của đề tài)." % q["avg_speed"]
        return reply

    # ---------- phạm vi thời gian 'từ X phút sang Y phút' ----------
    rng = re.search(r"(?:từ|tu)\s*(\d+)\s*(?:phút|phut|ph)\s*(?:sang|đến|den|den|đến|tới|toi)\s*(\d+)\s*(?:phút|phut|ph)?", m)
    if not rng:
        rng = re.search(r"(\d+)\s*[:\-]\s*(\d+)", m)  # "08:15" dạng giờ:phút video
    if rng:
        a, b = int(rng.group(1)), int(rng.group(2))
        if ":" in m or a > 10:                     # xem như phút (08:15 nghĩa là phút 8 đến 15)
            s, e = a * 60, b * 60
        else:
            s, e = a * 60, b * 60
        s, e = sorted([s, e])
        if e > video_sec_now + 1:
            e = int(video_sec_now)
        q = store.query_range(s, e)
        reply = ("🔎 **Phát %d–%d phút video** (%s): **%d xe**\n%s\nMật độ: **%.1f xe/phút**"
                 % (a, b, _fmt_range(s, e), q["total"], _counts_md(q["counts"], q["total"]), q["per_minute"]))
        if q["total"] > 0:
            reply += "\nTrạng thái đoạn này: **%s**" % q["status"]
        return reply + "\n\n" + _status_line(store)

    # ---------- 'dông không / mật độ / ùn tắc' ----------
    if "dông" in m or "tức" in m or "ùn tắc" in m or "un tac" in m or "mật độ" in m or "mat do" in m:
        q = store.query_range(0, video_sec_now)
        reply = ("🚦 Mật độ hiện tại: **%s**\n- Trong khung hình: **%d xe**\n- Trung bình 5 phút: **%d xe/phút**\n"
                 "- Tổng từ đầu: **%d xe** | Tốc độ TB: **%.1f km/h**"
                 % (st["status"], st["active_now"], st["per_minute"], st["total"], st["avg_speed"]))
        if st["status"].startswith("🔴") or st["status"].startswith("🟠"):
            reply += "\n⚠️ Đề xuất: hạn chế đèn xanh kéo dài, ưu tiên phương tiện công cộng qua nhánh này."
        return reply + "\n\n" + _status_line(store)

    # ---------- 'loai xe / xe may / xe tai / bus / oto' (co va khong dau) ----------
    cls_kw = {
        0: ("o to", "ô tô", "oto"),
        1: ("xe may", "xe máy"),
        2: ("xe tai", "xe tải"),
        3: ("xe bus", "bus"),
        4: ("xe ba giac", "xe ba gác", "ba giac", "ba gác"),
    }
    cls_hit = None
    for c in ALL_CLASSES:
        if any(k in m for k in cls_kw[c]):
            cls_hit = c
            break
    if cls_hit is not None:
        q = store.query_range(0, video_sec_now)
        n = q["counts"].get(cls_hit, 0)
        reply = ("🛵 **%s**: **%d chiếc** (%.1f%% trong tổng số %d xe từ đầu đến giờ).\n"
                 % (CLASS_NAMES[cls_hit], n,
                    (100.0 * n / q["total"] if q["total"] else 0), q["total"])
                 + _counts_md(q["counts"], q["total"]))
        return reply + "\n\n" + _status_line(store)

    # ---------- 'tốc độ' ----------
    if "tốc độ" in m or "toc do" in m or "nhanh" in m or "chậm" in m:
        q = store.query_range(0, video_sec_now)
        return ("⚡ **Tốc độ trung bình** (EMA α=0.2, ≥15 khung): **%.1f km/h** "
                "(hệ số pixel→mét 0.065).\nĐo từ quỹ đạo 30 điểm của ByteTrack.\n\n" + _status_line(store)) % q["avg_speed"]

    # ---------- mặc định ----------
    return ("Tôi chưa hiểu rõ câu hỏi — thử đổi cách hỏi, ví dụ:\n"
            "- *'nay có bao nhiêu xe?'*\n- *'từ 5 phút sang 10 phút có bao nhiêu xe máy?'*\n"
            "- *'mật độ thế nào / ùn tắc chưa?'*\n- *'tốc độ trung bình bao nhiêu?'*\n"
            "- *'/giúp đỡ*' để xem hết lệnh.\n\n" + _status_line(store))


def help_reply(store):
    return (
        "**Lệnh tôi hiểu được** (chạy song song với video, dữ liệu real-time):\n"
        "| Lệnh | Ví dụ |\n|---|---|\n"
        "| Tổng xe từ đầu đến giờ | `nay có bao nhiêu xe` |\n"
        "| Phạm vi phút video | `từ 5 phút sang 10 phút bao nhiêu xe` |\n"
        "| Theo loại xe | `có bao nhiêu xe máy` / `xe tải` / `bus` / `ô tô` / `xe ba gác` |\n"
        "| Mật độ / ùn tắc | `mật độ thế nào`, `ùn tắc chưa` |\n"
        "| Tốc độ | `tốc độ trung bình` |\n"
        "| Xuất CSV | `xuất csv` (thống kê theo phút, file sự kiện đã ghi liên tục) |\n\n"
        "_Dữ liệu: " + _status_line(store) + "_"
    )
