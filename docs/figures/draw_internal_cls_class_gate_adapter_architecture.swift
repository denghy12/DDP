import AppKit

let fixedGlobalAlpha = CommandLine.arguments.contains("--fixed-global-alpha")
let W: CGFloat = fixedGlobalAlpha ? 2600 : 1800
let H: CGFloat = fixedGlobalAlpha ? 1980 : 1200
let outputPath = fixedGlobalAlpha
    ? "docs/figures/code_ddp_internal_cls_fixed_global_alpha_architecture.png"
    : "docs/figures/code_ddp_internal_cls_class_gate_adapter_architecture.png"

func color(_ hex: UInt32, _ alpha: CGFloat = 1.0) -> NSColor {
    let r = CGFloat((hex >> 16) & 0xff) / 255.0
    let g = CGFloat((hex >> 8) & 0xff) / 255.0
    let b = CGFloat(hex & 0xff) / 255.0
    return NSColor(calibratedRed: r, green: g, blue: b, alpha: alpha)
}

func rectTop(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat) -> NSRect {
    NSRect(x: x, y: H - y - h, width: w, height: h)
}

func pointTop(_ x: CGFloat, _ y: CGFloat) -> NSPoint {
    NSPoint(x: x, y: H - y)
}

func drawText(
    _ text: String,
    in rect: NSRect,
    size: CGFloat = 22,
    weight: NSFont.Weight = .regular,
    color textColor: NSColor = .black,
    align: NSTextAlignment = .center
) {
    let paragraph = NSMutableParagraphStyle()
    paragraph.alignment = align
    paragraph.lineBreakMode = .byWordWrapping
    paragraph.lineSpacing = 2
    let attributes: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: size, weight: weight),
        .foregroundColor: textColor,
        .paragraphStyle: paragraph
    ]
    NSAttributedString(string: text, attributes: attributes)
        .draw(in: rect.insetBy(dx: 8, dy: 2))
}

func fillRoundRect(
    _ rect: NSRect,
    radius: CGFloat,
    fill: NSColor,
    stroke: NSColor,
    lineWidth: CGFloat = 2
) {
    let path = NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius)
    fill.setFill()
    path.fill()
    stroke.setStroke()
    path.lineWidth = lineWidth
    path.stroke()
}

func box(
    _ x: CGFloat,
    _ y: CGFloat,
    _ w: CGFloat,
    _ h: CGFloat,
    _ title: String,
    _ body: String = "",
    stroke: NSColor = color(0xb8c4d8),
    fill: NSColor = color(0xffffff, 0.94),
    titleColor: NSColor = .black,
    titleSize: CGFloat = 21,
    bodySize: CGFloat = 15,
    titleFraction: CGFloat = 0.38,
    radius: CGFloat = 8
) {
    let rect = rectTop(x, y, w, h)
    fillRoundRect(rect, radius: radius, fill: fill, stroke: stroke, lineWidth: 1.8)
    let titleHeight = h * titleFraction
    drawText(
        title,
        in: NSRect(
            x: rect.minX + 4,
            y: rect.maxY - titleHeight - 4,
            width: rect.width - 8,
            height: titleHeight
        ),
        size: titleSize,
        weight: .semibold,
        color: titleColor
    )
    if !body.isEmpty {
        drawText(
            body,
            in: NSRect(
                x: rect.minX + 4,
                y: rect.minY + 5,
                width: rect.width - 8,
                height: h - titleHeight - 8
            ),
            size: bodySize,
            weight: .regular,
            color: color(0x111827)
        )
    }
}

func panel(
    _ x: CGFloat,
    _ y: CGFloat,
    _ w: CGFloat,
    _ h: CGFloat,
    fill: NSColor,
    stroke: NSColor,
    title: String,
    titleColor: NSColor,
    centered: Bool = false
) {
    let rect = rectTop(x, y, w, h)
    fillRoundRect(rect, radius: 10, fill: fill, stroke: stroke, lineWidth: 2.2)
    drawText(
        title,
        in: rectTop(x + 16, y + 8, w - 32, 36),
        size: 24,
        weight: .bold,
        color: titleColor,
        align: centered ? .center : .left
    )
}

func arrowHead(
    from startTuple: (CGFloat, CGFloat),
    to endTuple: (CGFloat, CGFloat),
    color: NSColor
) {
    let start = pointTop(startTuple.0, startTuple.1)
    let end = pointTop(endTuple.0, endTuple.1)
    let angle = atan2(end.y - start.y, end.x - start.x)
    let length: CGFloat = 15
    let spread: CGFloat = .pi / 7
    let p1 = NSPoint(
        x: end.x - length * cos(angle - spread),
        y: end.y - length * sin(angle - spread)
    )
    let p2 = NSPoint(
        x: end.x - length * cos(angle + spread),
        y: end.y - length * sin(angle + spread)
    )
    let head = NSBezierPath()
    head.move(to: end)
    head.line(to: p1)
    head.line(to: p2)
    head.close()
    color.setFill()
    head.fill()
}

func orthogonalArrow(
    _ points: [(CGFloat, CGFloat)],
    color: NSColor,
    width: CGFloat = 4
) {
    guard points.count >= 2 else { return }
    for index in 1..<points.count {
        let a = points[index - 1]
        let b = points[index]
        precondition(
            a.0 == b.0 || a.1 == b.1,
            "Every connector segment must be horizontal or vertical"
        )
    }
    let path = NSBezierPath()
    path.move(to: pointTop(points[0].0, points[0].1))
    for point in points.dropFirst() {
        path.line(to: pointTop(point.0, point.1))
    }
    color.setStroke()
    path.lineWidth = width
    path.lineCapStyle = .round
    path.lineJoinStyle = .round
    path.stroke()
    arrowHead(
        from: points[points.count - 2],
        to: points[points.count - 1],
        color: color
    )
}

func legendItem(
    _ x: CGFloat,
    _ y: CGFloat,
    _ stroke: NSColor,
    _ fill: NSColor,
    _ label: String,
    width: CGFloat
) {
    fillRoundRect(rectTop(x, y, 28, 24), radius: 5, fill: fill, stroke: stroke, lineWidth: 2)
    drawText(
        label,
        in: rectTop(x + 38, y - 3, width, 32),
        size: 15,
        weight: .regular,
        color: color(0x111827),
        align: .left
    )
}

let blue = color(0x2563eb)
let red = color(0xdc2626)
let green = color(0x059669)
let orange = color(0xf97316)
let purple = color(0x7c3aed)
let slate = color(0x475569)

func drawDetailedFixedGlobalAlpha() {
    color(0xf8fbff).setFill()
    NSRect(x: 0, y: 0, width: W, height: H).fill()

    drawText(
        "CODE_DDP: EMOTIC B5-C3 Internal CLS Adapter (Fixed Global α)",
        in: rectTop(0, 4, W, 54),
        size: 38,
        weight: .bold,
        color: color(0x0f172a)
    )
    drawText(
        "原始 DDP token-attention pooled 主路 + 独立 CLS residual logit-correction 支路",
        in: rectTop(0, 52, W, 30),
        size: 18,
        weight: .semibold,
        color: slate
    )

    // ------------------------------------------------------------------
    // Feature construction: image and text branches.
    // ------------------------------------------------------------------
    panel(
        20, 88, 1490, 630,
        fill: color(0xeef6ff, 0.94),
        stroke: blue,
        title: "图像分支：class-specific 正 / 负 visual paths",
        titleColor: color(0x1d4ed8)
    )
    panel(
        1530, 88, 850, 630,
        fill: color(0xfff1f2, 0.94),
        stroke: red,
        title: "文本分支：class-specific 正 / 负语义特征",
        titleColor: color(0xb91c1c)
    )

    box(70, 155, 360, 90, "输入图像 x", "B × 3 × 224 × 224")
    box(570, 155, 350, 90, "当前 task 的 seen classes", "K = 5, 8, …, 26")
    box(
        1040, 150, 400, 100,
        "正 / 负 visual prompts",
        "每类 Vc− / Vc+；16 × 768\n总计 2K 组 class-specific paths",
        stroke: orange,
        fill: color(0xfffbeb),
        titleColor: color(0x9a3412),
        titleSize: 20,
        bodySize: 14
    )
    box(
        70, 305, 380, 105,
        "图像复制为 2K 条路径",
        "B → 2KB\n顺序：[K 条 negative paths, K 条 positive paths]",
        titleSize: 20,
        bodySize: 14
    )
    box(
        500, 335, 830, 125,
        "冻结 CLIP ViT-B/16 Visual Encoder",
        "visual prompts 注入第 7–11 层 Q/K/V；12-layer Transformer\n每条路径输出 197 个 512-d token，并逐 token L2 normalize",
        stroke: blue,
        fill: color(0xf8fbff),
        titleColor: color(0x1d4ed8),
        titleSize: 23,
        bodySize: 15
    )
    box(
        520, 555, 790, 105,
        "共享 encoder 输出：完整 token features  Z",
        "代码张量 B × 2K × 512 × 197\n同一个 Z 向下分成：全部 197 tokens 主路 与 index 0 CLS 支路",
        stroke: color(0x64748b),
        fill: color(0xffffff),
        titleSize: 21,
        bodySize: 14
    )

    orthogonalArrow([(250, 245), (250, 305)], color: blue)
    orthogonalArrow([(920, 200), (1040, 200)], color: blue)
    orthogonalArrow([(450, 357), (475, 357), (475, 397), (500, 397)], color: blue)
    orthogonalArrow([(1240, 250), (1240, 300), (1140, 300), (1140, 335)], color: blue)
    orthogonalArrow([(915, 460), (915, 555)], color: blue)

    box(1740, 150, 430, 86, "EMOTIC 类别", "26 类；B5-C3 共 8 个 tasks")
    box(
        1570, 275, 360, 94,
        "Negative prompt",
        "a photo of a person not feeling {class}",
        stroke: red,
        fill: color(0xfffbfb),
        titleColor: color(0xb91c1c),
        titleSize: 19,
        bodySize: 12
    )
    box(
        1980, 275, 360, 94,
        "Positive prompt",
        "a photo of a person clearly feeling {class}",
        stroke: red,
        fill: color(0xfffbfb),
        titleColor: color(0xb91c1c),
        titleSize: 19,
        bodySize: 12
    )
    box(
        1690, 405, 520, 86,
        "CSC Prompt Learner",
        "正 / 负 context：7 tokens × 768；每类独立语义 context",
        stroke: orange,
        fill: color(0xfffbeb),
        titleColor: color(0x9a3412),
        titleSize: 20,
        bodySize: 14
    )
    box(
        1690, 515, 520, 76,
        "冻结 CLIP Text Encoder",
        "Transformer + text projection；L2 normalize",
        stroke: red,
        fill: color(0xfffbfb),
        titleColor: color(0xb91c1c),
        titleSize: 20,
        bodySize: 14
    )
    box(
        1740, 615, 420, 72,
        "文本特征 E− / E+",
        "2K × 512；与 2K visual paths 一一配对",
        stroke: color(0x64748b),
        fill: color(0xffffff),
        titleSize: 19,
        bodySize: 13
    )

    orthogonalArrow([(1955, 236), (1955, 255), (1750, 255), (1750, 275)], color: red)
    orthogonalArrow([(1955, 236), (1955, 255), (2160, 255), (2160, 275)], color: red)
    orthogonalArrow([(1750, 369), (1750, 388), (1950, 388), (1950, 405)], color: red)
    orthogonalArrow([(2160, 369), (2160, 388), (1950, 388), (1950, 405)], color: red)
    orthogonalArrow([(1950, 491), (1950, 515)], color: red)
    orthogonalArrow([(1950, 591), (1950, 615)], color: red)

    // ------------------------------------------------------------------
    // Original DDP main path. It uses all tokens and text-conditioned
    // attention pooling; it is deliberately distinct from the CLS path.
    // ------------------------------------------------------------------
    panel(
        20, 745, 2360, 430,
        fill: color(0xf0fdf4, 0.96),
        stroke: green,
        title: "原始 DDP 主路：文本条件 token similarity → 正路径 token attention → pooled path feature → base logits",
        titleColor: color(0x047857),
        centered: true
    )

    box(
        85, 805, 275, 48,
        "输入端口：同一图像分支 Z",
        "",
        stroke: blue,
        fill: color(0xf8fbff),
        titleColor: color(0x1d4ed8),
        titleSize: 17,
        titleFraction: 0.9
    )
    box(
        465, 805, 280, 48,
        "输入端口：同一文本分支 E− / E+",
        "",
        stroke: red,
        fill: color(0xfffbfb),
        titleColor: color(0xb91c1c),
        titleSize: 16,
        titleFraction: 0.9
    )

    box(
        85, 875, 275, 155,
        "使用全部 197 tokens",
        "Zp = {zp,n} n=1…197\np ∈ {c−, c+}\n不是 CLS feature",
        stroke: blue,
        fill: color(0xf8fbff),
        titleColor: color(0x1d4ed8),
        titleSize: 21,
        bodySize: 15
    )
    box(
        410, 865, 390, 175,
        "① Token-wise text similarity",
        "Sp,n = 20 · 〈zp,n, Ep〉\nE− 对应 negative path\nE+ 对应 positive path\nS: B × 2K × 197",
        stroke: red,
        fill: color(0xfffbfb),
        titleColor: color(0xb91c1c),
        titleSize: 21,
        bodySize: 15,
        titleFraction: 0.25
    )
    box(
        850, 865, 355, 175,
        "② 正路径产生 token attention",
        "wc = softmaxn(Sc+,n)\n同一个 wc 复制给 c− 与 c+\npath weights: [w, w]",
        stroke: green,
        fill: color(0xffffff),
        titleColor: color(0x047857),
        titleSize: 20,
        bodySize: 15,
        titleFraction: 0.28
    )
    box(
        1250, 865, 360, 175,
        "③ DDP pooled path features",
        "z̄c± = Σn wc,n · Zc±,n\nB × 2K × 512\n文本引导的全 token 加权池化\n与 zCLS 是两种不同 feature",
        stroke: green,
        fill: color(0xffffff),
        titleColor: color(0x047857),
        titleSize: 20,
        bodySize: 14,
        titleFraction: 0.25
    )
    box(
        1625, 865, 390, 175,
        "④ 原始 DDP 正 / 负 path logits",
        "lDDP,p = 5 · Σn(Sp,n wc,n)\n等价于 100 · 〈z̄p, Ep〉\n输出 B × 2K；随后 reshape B × 2 × K",
        stroke: green,
        fill: color(0xffffff),
        titleColor: color(0x047857),
        titleSize: 20,
        bodySize: 14,
        titleFraction: 0.25
    )
    box(
        2060, 882, 240, 140,
        "主路输出",
        "LDDP = [l−, l+]\n保留为恒等基线\n不经过 Adapter",
        stroke: blue,
        fill: color(0xf8fbff),
        titleColor: color(0x1d4ed8),
        titleSize: 21,
        bodySize: 15
    )

    // Local shared-tensor ports replace cross-panel wires.
    orthogonalArrow([(222, 853), (222, 875)], color: blue)
    orthogonalArrow([(605, 853), (605, 865)], color: red)

    orthogonalArrow([(360, 952), (410, 952)], color: green)
    orthogonalArrow([(800, 952), (850, 952)], color: green)
    orthogonalArrow([(1205, 952), (1250, 952)], color: green)
    orthogonalArrow([(1610, 952), (1625, 952)], color: green)
    orthogonalArrow([(2015, 952), (2060, 952)], color: blue)

    // ------------------------------------------------------------------
    // Internal CLS residual branch. This skips DDP attention pooling and
    // expands the 512 -> 128 -> 512 Adapter into explicit operations.
    // ------------------------------------------------------------------
    panel(
        20, 1200, 2360, 470,
        fill: color(0xfffbeb, 0.72),
        stroke: orange,
        title: "内部 CLS correction 支路：直接取 index-0 token；共享 Adapter；固定全局 α=0.03；校正原始 DDP logits",
        titleColor: color(0x9a3412),
        centered: false
    )

    box(
        55, 1272, 255, 50,
        "输入端口：同一 Z 的 index-0 token",
        "",
        stroke: blue,
        fill: color(0xf8fbff),
        titleColor: color(0x1d4ed8),
        titleSize: 15,
        titleFraction: 0.9
    )
    box(
        1690, 1272, 300, 50,
        "输入端口：同一文本特征 E− / E+",
        "",
        stroke: red,
        fill: color(0xfffbfb),
        titleColor: color(0xb91c1c),
        titleSize: 16,
        titleFraction: 0.9
    )

    box(
        55, 1345, 255, 145,
        "独立 CLS path feature",
        "zpCLS = Zp[:,:,0]\nB × 2K × 512\n不使用 wc\n不等于 DDP pooled z̄p",
        stroke: blue,
        fill: color(0xf8fbff),
        titleColor: color(0x1d4ed8),
        titleSize: 20,
        bodySize: 14,
        titleFraction: 0.26
    )

    let adapterRect = rectTop(345, 1305, 1025, 215)
    let adapterPath = NSBezierPath(roundedRect: adapterRect, xRadius: 12, yRadius: 12)
    adapterPath.setLineDash([9, 6], count: 2, phase: 0)
    color(0xffffff, 0.72).setFill()
    adapterPath.fill()
    orange.setStroke()
    adapterPath.lineWidth = 2
    adapterPath.stroke()
    drawText(
        "Shared Residual Adapter · Base5 train 16-shot · 3 seeds · 迁移后冻结 · 131,072 参数",
        in: rectTop(365, 1310, 985, 34),
        size: 18,
        weight: .semibold,
        color: color(0x9a3412)
    )

    box(
        370, 1370, 165, 105,
        "Normalize",
        "ẑp = zpCLS / ‖zpCLS‖₂\n512-d",
        stroke: orange,
        fill: color(0xffffff),
        titleColor: color(0x9a3412),
        titleSize: 19,
        bodySize: 13
    )
    box(
        560, 1370, 175, 105,
        "W↓ Down projection",
        "512 → 128\nno bias",
        stroke: orange,
        fill: color(0xffffff),
        titleColor: color(0x9a3412),
        titleSize: 17,
        bodySize: 13
    )
    box(
        760, 1370, 140, 105,
        "GELU",
        "非线性激活\n128-d",
        stroke: orange,
        fill: color(0xffffff),
        titleColor: color(0x9a3412),
        titleSize: 19,
        bodySize: 13
    )
    box(
        925, 1370, 175, 105,
        "W↑ Up projection",
        "128 → 512\nno bias",
        stroke: orange,
        fill: color(0xffffff),
        titleColor: color(0x9a3412),
        titleSize: 17,
        bodySize: 13
    )
    box(
        1125, 1370, 220, 105,
        "固定全局 α=0.03",
        "rp = W↑ GELU(W↓ẑp)\nΔzp = α · rp",
        stroke: purple,
        fill: color(0xf5f3ff),
        titleColor: color(0x6d28d9),
        titleSize: 19,
        bodySize: 13
    )
    box(
        1410, 1345, 240, 145,
        "Adapter feature residual",
        "Aα(zp)=zpCLS+Δzp\nΔzp=Aα(zp)−zpCLS\n这里只保留差值 Δzp\n不替换 DDP pooled feature",
        stroke: orange,
        fill: color(0xffffff),
        titleColor: color(0x9a3412),
        titleSize: 18,
        bodySize: 13,
        titleFraction: 0.25
    )
    box(
        1690, 1345, 300, 145,
        "与同一文本特征做校正相似度",
        "Δlp = 100 · 〈Δzp, Ep〉\nE− 校正 negative path\nE+ 校正 positive path\n输出 B × 2K",
        stroke: red,
        fill: color(0xfffbfb),
        titleColor: color(0xb91c1c),
        titleSize: 18,
        bodySize: 13,
        titleFraction: 0.28
    )
    box(
        2030, 1345, 300, 160,
        "固定全局 α 的最终预测",
        "lfinal,p = lDDP,p + Δlp\np = softmax([lfinal,−, lfinal,+]/T)positive\nFinal Test mAP: 31.3129 ± 0.2036",
        stroke: purple,
        fill: color(0xf5f3ff),
        titleColor: color(0x6d28d9),
        titleSize: 19,
        bodySize: 13,
        titleFraction: 0.28
    )

    // Short, non-crossing local connectors. The main path output is aligned
    // directly over the final add/softmax box.
    orthogonalArrow([(182, 1322), (182, 1345)], color: blue)
    orthogonalArrow([(310, 1417), (370, 1417)], color: orange)
    orthogonalArrow([(535, 1422), (560, 1422)], color: orange)
    orthogonalArrow([(735, 1422), (760, 1422)], color: orange)
    orthogonalArrow([(900, 1422), (925, 1422)], color: orange)
    orthogonalArrow([(1100, 1422), (1125, 1422)], color: orange)
    orthogonalArrow([(1345, 1422), (1410, 1422)], color: orange)
    orthogonalArrow([(1650, 1417), (1690, 1417)], color: orange)
    orthogonalArrow([(1840, 1322), (1840, 1345)], color: red)
    orthogonalArrow([(1990, 1417), (2030, 1417)], color: purple)
    orthogonalArrow([(2180, 1022), (2180, 1345)], color: blue)

    // Legend / protocol strip.
    let strip = rectTop(20, 1690, 2360, 92)
    let dashed = NSBezierPath(roundedRect: strip, xRadius: 9, yRadius: 9)
    dashed.setLineDash([8, 6], count: 2, phase: 0)
    color(0xffffff, 0.82).setFill()
    dashed.fill()
    color(0x94a3b8).setStroke()
    dashed.lineWidth = 1.6
    dashed.stroke()

    legendItem(55, 1722, blue, color(0xf8fbff), "重复输入端口代表同一 Z / E 张量，不是第二次编码", width: 440)
    legendItem(620, 1722, orange, color(0xfffbeb), "Adapter：512→128→512；共享且迁移后冻结", width: 420)
    legendItem(1140, 1722, red, color(0xfffbfb), "同一 E−/E+ 同时服务两条相似度支路", width: 390)
    legendItem(1630, 1722, purple, color(0xf5f3ff), "α=0.03 对所有 task / class 固定；无 gate / βc / 外部分数融合", width: 650)
}

func drawSideBySideFixedGlobalAlpha() {
    color(0xf8fbff).setFill()
    NSRect(x: 0, y: 0, width: W, height: H).fill()

    drawText(
        "CODE_DDP: EMOTIC B5-C3 Internal CLS Adapter (Fixed Global α)",
        in: rectTop(0, 2, W, 55),
        size: 40,
        weight: .bold,
        color: color(0x0f172a)
    )
    drawText(
        "共享 CLIP 编码 → 两种图像 feature → 左右双分路 → 独立 logit residual 汇合",
        in: rectTop(0, 54, W, 30),
        size: 19,
        weight: .semibold,
        color: slate
    )

    // ================================================================
    // Shared feature generation.
    // ================================================================
    panel(
        20, 90, 1530, 500,
        fill: color(0xeef6ff, 0.94),
        stroke: blue,
        title: "图像分支：一次 class-specific DDP visual encoding，显式生成两种 feature",
        titleColor: color(0x1d4ed8)
    )
    panel(
        1570, 90, 1010, 500,
        fill: color(0xfff1f2, 0.94),
        stroke: red,
        title: "文本分支：共享正 / 负语义特征 E− / E+",
        titleColor: color(0xb91c1c)
    )

    box(65, 160, 290, 82, "输入图像 x", "B × 3 × 224 × 224", titleSize: 22, bodySize: 16)
    box(410, 160, 290, 82, "当前 seen classes", "K = 5, 8, …, 26", titleSize: 21, bodySize: 16)
    box(
        755, 150, 330, 102,
        "正 / 负 visual prompts",
        "每类 Vc− / Vc+；16 × 768\n图像复制为 2K 条 class-specific paths",
        stroke: orange,
        fill: color(0xfffbeb),
        titleColor: color(0x9a3412),
        titleSize: 20,
        bodySize: 14
    )
    box(
        1135, 145, 360, 112,
        "冻结 CLIP ViT-B/16 Visual Encoder",
        "visual prompts 注入第 7–11 层 Q/K/V\n输出 B × 2K × 512 × 197；逐 token normalize",
        stroke: blue,
        fill: color(0xf8fbff),
        titleColor: color(0x1d4ed8),
        titleSize: 19,
        bodySize: 14
    )
    box(
        450, 305, 650, 85,
        "共享 token tensor Z",
        "每条正 / 负路径含 197 个 512-d tokens；仅执行一次 DDP visual encoding",
        stroke: color(0x64748b),
        fill: color(0xffffff),
        titleSize: 21,
        bodySize: 14
    )
    box(
        105, 445, 590, 105,
        "图像输出 ①：完整 token features  Ztokens",
        "Ztokens = Z[:,:,:,:]；B × 2K × 512 × 197\n送入左侧原始 DDP token-attention / pooling 主路",
        stroke: blue,
        fill: color(0xf8fbff),
        titleColor: color(0x1d4ed8),
        titleSize: 20,
        bodySize: 15
    )
    box(
        830, 445, 590, 105,
        "图像输出 ②：独立 CLS path features  zCLS",
        "zCLS = Z[:,:,:,0]；B × 2K × 512\n跳过 token attention，送入右侧共享 Adapter 分路",
        stroke: blue,
        fill: color(0xf8fbff),
        titleColor: color(0x1d4ed8),
        titleSize: 20,
        bodySize: 15
    )

    orthogonalArrow([(355, 201), (410, 201)], color: blue)
    orthogonalArrow([(700, 201), (755, 201)], color: blue)
    orthogonalArrow([(1085, 201), (1135, 201)], color: blue)
    orthogonalArrow([(1315, 257), (1315, 285), (775, 285), (775, 305)], color: blue)
    orthogonalArrow([(650, 390), (650, 420), (400, 420), (400, 445)], color: blue)
    orthogonalArrow([(900, 390), (900, 420), (1125, 420), (1125, 445)], color: blue)

    box(1840, 145, 470, 75, "EMOTIC 类别", "26 类；B5-C3 共 8 个 tasks", titleSize: 22, bodySize: 15)
    box(
        1605, 260, 420, 92,
        "Negative semantic prompt",
        "a photo of a person not feeling {class}",
        stroke: red,
        fill: color(0xfffbfb),
        titleColor: color(0xb91c1c),
        titleSize: 19,
        bodySize: 13
    )
    box(
        2125, 260, 420, 92,
        "Positive semantic prompt",
        "a photo of a person clearly feeling {class}",
        stroke: red,
        fill: color(0xfffbfb),
        titleColor: color(0xb91c1c),
        titleSize: 19,
        bodySize: 13
    )
    box(
        1760, 390, 630, 82,
        "CSC Prompt Learner → 冻结 CLIP Text Encoder",
        "7 learnable context tokens / sign / class；text projection；L2 normalize",
        stroke: orange,
        fill: color(0xfffbeb),
        titleColor: color(0x9a3412),
        titleSize: 19,
        bodySize: 14
    )
    box(
        1840, 495, 470, 70,
        "文本输出：E− / E+",
        "2K × 512；与 2K visual paths 一一配对",
        stroke: red,
        fill: color(0xfffbfb),
        titleColor: color(0xb91c1c),
        titleSize: 20,
        bodySize: 14
    )

    orthogonalArrow([(2075, 220), (2075, 240), (1815, 240), (1815, 260)], color: red)
    orthogonalArrow([(2075, 220), (2075, 240), (2335, 240), (2335, 260)], color: red)
    orthogonalArrow([(1815, 352), (1815, 372), (2075, 372), (2075, 390)], color: red)
    orthogonalArrow([(2335, 352), (2335, 372), (2075, 372), (2075, 390)], color: red)
    orthogonalArrow([(2075, 472), (2075, 495)], color: red)

    // Explicit branch-output ports. These are not re-encodings: repeated E
    // boxes denote the exact same cached text tensor routed to both towers.
    drawText(
        "上方分支 → 模型的显式输入端口（重复 E 端口表示同一张量，不是第二次 text encoding）",
        in: rectTop(20, 602, 2560, 32),
        size: 18,
        weight: .semibold,
        color: slate
    )
    box(
        120, 638, 430, 78,
        "Image → DDP：Ztokens",
        "B × 2K × 512 × 197",
        stroke: blue,
        fill: color(0xf8fbff),
        titleColor: color(0x1d4ed8),
        titleSize: 20,
        bodySize: 14
    )
    box(
        730, 638, 430, 78,
        "Text → DDP：E− / E+",
        "2K × 512；用于 token-wise similarity",
        stroke: red,
        fill: color(0xfffbfb),
        titleColor: color(0xb91c1c),
        titleSize: 20,
        bodySize: 14
    )
    box(
        1435, 638, 430, 78,
        "Image → Adapter：zCLS",
        "B × 2K × 512；index-0 token",
        stroke: blue,
        fill: color(0xf8fbff),
        titleColor: color(0x1d4ed8),
        titleSize: 20,
        bodySize: 14
    )
    box(
        2045, 638, 430, 78,
        "Text → Adapter：同一 E− / E+",
        "2K × 512；用于 correction similarity",
        stroke: red,
        fill: color(0xfffbfb),
        titleColor: color(0xb91c1c),
        titleSize: 19,
        bodySize: 14
    )

    // ================================================================
    // Two vertical towers, side by side.
    // ================================================================
    panel(
        20, 755, 1250, 900,
        fill: color(0xf0fdf4, 0.96),
        stroke: green,
        title: "左塔：原始 DDP pooled / token-attention 主路",
        titleColor: color(0x047857),
        centered: true
    )
    panel(
        1330, 755, 1250, 900,
        fill: color(0xfffbeb, 0.74),
        stroke: orange,
        title: "右塔：Internal CLS Shared Residual Adapter 分路",
        titleColor: color(0x9a3412),
        centered: true
    )

    box(
        270, 855, 750, 125,
        "① Token-wise text similarity（图像与文本第一次明确汇合）",
        "Sp,n = 20 · 〈Zp,n, Ep〉；p ∈ {c−,c+}；n=1…197\nE− 对应 negative path，E+ 对应 positive path；S: B × 2K × 197",
        stroke: red,
        fill: color(0xfffbfb),
        titleColor: color(0xb91c1c),
        titleSize: 21,
        bodySize: 15
    )
    box(
        270, 1025, 750, 120,
        "② Positive-path token attention",
        "wc = softmaxn(Sc+,n)；只用 positive path 产生权重\n同一个 wc 复制给该类别的 c− / c+ 两条路径：path weights = [w,w]",
        stroke: green,
        fill: color(0xffffff),
        titleColor: color(0x047857),
        titleSize: 21,
        bodySize: 15
    )
    box(
        270, 1190, 750, 125,
        "③ DDP pooled path feature（不同于 CLS feature）",
        "z̄c± = Σn wc,n · Zc±,n；B × 2K × 512\n它是 E+ 引导的全 197-token 加权池化结果，不是 Z[:,:,:,0]",
        stroke: green,
        fill: color(0xffffff),
        titleColor: color(0x047857),
        titleSize: 21,
        bodySize: 15
    )
    box(
        270, 1360, 750, 125,
        "④ 原始 DDP positive / negative path logits",
        "lDDP,p = 5 · Σn(Sp,n wc,n) = 100 · 〈z̄p, Ep〉\n输出 B × 2K，reshape 为 B × 2 × K；完全保留原始 DDP 主路",
        stroke: green,
        fill: color(0xffffff),
        titleColor: color(0x047857),
        titleSize: 21,
        bodySize: 15
    )
    box(
        390, 1525, 510, 95,
        "左塔输出：LDDP = [lDDP,−, lDDP,+]",
        "原始 task logits；不经过 Adapter",
        stroke: blue,
        fill: color(0xf8fbff),
        titleColor: color(0x1d4ed8),
        titleSize: 21,
        bodySize: 15
    )

    orthogonalArrow([(335, 716), (335, 805), (480, 805), (480, 855)], color: blue)
    orthogonalArrow([(945, 716), (945, 805), (810, 805), (810, 855)], color: red)
    orthogonalArrow([(645, 980), (645, 1025)], color: green)
    orthogonalArrow([(645, 1145), (645, 1190)], color: green)
    orthogonalArrow([(645, 1315), (645, 1360)], color: green)
    orthogonalArrow([(645, 1485), (645, 1525)], color: blue)

    // Adapter tower: explicit 512 -> 128 -> 512 operations.
    let towerAdapterRect = rectTop(1395, 845, 610, 745)
    let towerAdapterPath = NSBezierPath(roundedRect: towerAdapterRect, xRadius: 12, yRadius: 12)
    towerAdapterPath.setLineDash([9, 6], count: 2, phase: 0)
    color(0xffffff, 0.70).setFill()
    towerAdapterPath.fill()
    orange.setStroke()
    towerAdapterPath.lineWidth = 2
    towerAdapterPath.stroke()
    drawText(
        "Shared Adapter W：Base5 train 16-shot · 3 seeds · 迁移后冻结 · 131,072 参数",
        in: rectTop(1415, 855, 570, 40),
        size: 18,
        weight: .semibold,
        color: color(0x9a3412)
    )
    box(
        1455, 925, 490, 90,
        "① 独立 CLS path feature",
        "zpCLS = Zp[:,:,:,0]；B × 2K × 512；不使用 wc，不经过 DDP pooling",
        stroke: blue,
        fill: color(0xf8fbff),
        titleColor: color(0x1d4ed8),
        titleSize: 20,
        bodySize: 14
    )
    box(
        1455, 1050, 490, 86,
        "② Normalize",
        "ẑp = zpCLS / ‖zpCLS‖₂；512-d",
        stroke: orange,
        fill: color(0xffffff),
        titleColor: color(0x9a3412),
        titleSize: 20,
        bodySize: 14
    )
    box(
        1455, 1170, 490, 86,
        "③ W↓ → GELU → W↑",
        "512 → 128 → 512；down / up 均无 bias",
        stroke: orange,
        fill: color(0xffffff),
        titleColor: color(0x9a3412),
        titleSize: 20,
        bodySize: 14
    )
    box(
        1455, 1290, 490, 105,
        "④ 固定全局 α=0.03 与 feature residual",
        "rp = W↑ GELU(W↓ẑp)；Δzp = αrp\nAα(zp)=zpCLS+Δzp；Δzp=Aα(zp)−zpCLS",
        stroke: purple,
        fill: color(0xf5f3ff),
        titleColor: color(0x6d28d9),
        titleSize: 19,
        bodySize: 14
    )
    box(
        2070, 1190, 440, 180,
        "⑤ Correction similarity（图像与文本第二次明确汇合）",
        "Δlp = 100 · 〈Δzp, Ep〉\nE− 校正 negative path\nE+ 校正 positive path\n输出 B × 2K；不替换 DDP pooled feature",
        stroke: red,
        fill: color(0xfffbfb),
        titleColor: color(0xb91c1c),
        titleSize: 19,
        bodySize: 15,
        titleFraction: 0.28
    )
    box(
        2070, 1450, 440, 110,
        "右塔输出：ΔL = [Δl−, Δl+]",
        "仅提供 CLS residual logit correction",
        stroke: orange,
        fill: color(0xffffff),
        titleColor: color(0x9a3412),
        titleSize: 20,
        bodySize: 15
    )

    orthogonalArrow([(1650, 716), (1650, 925)], color: blue)
    orthogonalArrow([(1700, 1015), (1700, 1050)], color: orange)
    orthogonalArrow([(1700, 1136), (1700, 1170)], color: orange)
    orthogonalArrow([(1700, 1256), (1700, 1290)], color: orange)
    orthogonalArrow([(1945, 1342), (2010, 1342), (2010, 1280), (2070, 1280)], color: orange)
    orthogonalArrow([(2260, 716), (2260, 1125), (2290, 1125), (2290, 1190)], color: red)
    orthogonalArrow([(2290, 1370), (2290, 1450)], color: orange)

    // ================================================================
    // Independent fusion/result node outside both towers.
    // ================================================================
    box(
        835, 1715, 930, 155,
        "独立 Logit Residual Add + Fixed-global-α 最终预测",
        "lfinal,p = lDDP,p + Δlp；reshape B × 2 × K\np = softmax([lfinal,−, lfinal,+] / T)positive；所有 task / class 固定 α=0.03\n无 class gate、无 βc、无外部模型分数融合；Final Test mAP: 31.3129 ± 0.2036",
        stroke: purple,
        fill: color(0xf5f3ff),
        titleColor: color(0x6d28d9),
        titleSize: 23,
        bodySize: 16,
        titleFraction: 0.28
    )
    orthogonalArrow([(645, 1620), (645, 1792), (835, 1792)], color: blue)
    orthogonalArrow([(2290, 1560), (2290, 1792), (1765, 1792)], color: orange)

    drawText(
        "LDDP（原始 DDP 主路）",
        in: rectTop(625, 1735, 200, 32),
        size: 16,
        weight: .semibold,
        color: color(0x1d4ed8)
    )
    drawText(
        "ΔL（CLS Adapter 分路）",
        in: rectTop(1775, 1735, 230, 32),
        size: 16,
        weight: .semibold,
        color: color(0x9a3412)
    )

    let strip = rectTop(20, 1892, 2560, 72)
    let dashed = NSBezierPath(roundedRect: strip, xRadius: 9, yRadius: 9)
    dashed.setLineDash([8, 6], count: 2, phase: 0)
    color(0xffffff, 0.84).setFill()
    dashed.fill()
    color(0x94a3b8).setStroke()
    dashed.lineWidth = 1.6
    dashed.stroke()
    legendItem(55, 1916, blue, color(0xf8fbff), "冻结 DDP / CLIP；蓝色为图像 feature 与原始 logits", width: 470)
    legendItem(650, 1916, red, color(0xfffbfb), "同一 E−/E+ 同时进入左塔与右塔", width: 380)
    legendItem(1120, 1916, orange, color(0xfffbeb), "Adapter 共享且冻结；仅产生 ΔL", width: 360)
    legendItem(1570, 1916, purple, color(0xf5f3ff), "紫色最终预测框独立于两塔；只在 logits 层汇合", width: 600)
}

let image = NSImage(size: NSSize(width: W, height: H))
image.lockFocus()

if fixedGlobalAlpha {
    drawSideBySideFixedGlobalAlpha()
} else {
color(0xf8fbff).setFill()
NSRect(x: 0, y: 0, width: W, height: H).fill()

drawText(
    fixedGlobalAlpha
        ? "CODE_DDP: EMOTIC B5-C3 Internal CLS Adapter (Fixed Global α)"
        : "CODE_DDP: EMOTIC B5-C3 Internal CLS Adapter + Task Alpha + Class Gate",
    in: rectTop(0, 0, W, 48),
    size: 34,
    weight: .bold,
    color: color(0x0f172a)
)

panel(
    14, 56, 1038, 585,
    fill: color(0xeef6ff, 0.92),
    stroke: blue,
    title: "图像分支：class-specific dual visual paths + CLS feature",
    titleColor: color(0x1d4ed8)
)
panel(
    1068, 56, 717, 585,
    fill: color(0xfff1f2, 0.92),
    stroke: red,
    title: "文本分支：class-specific semantic contexts",
    titleColor: color(0xb91c1c)
)
panel(
    14, 660, 1771, 430,
    fill: color(0xecfdf5, 0.96),
    stroke: green,
    title: fixedGlobalAlpha
        ? ""
        : "DDP 内部共享 CLS Adapter：一次 DDP 编码；val 选择；test 只评估",
    titleColor: color(0x047857),
    centered: true
)

if fixedGlobalAlpha {
    // Keep the section title in the clear lane between the three descending
    // feature connectors so no arrow runs through the label.
    drawText(
        "DDP 内部共享 CLS Adapter · 固定全局 α=0.03 · 无逐类门控",
        in: rectTop(640, 668, 650, 34),
        size: 22,
        weight: .bold,
        color: color(0x047857)
    )
}

// Image branch.
box(55, 118, 290, 78, "输入图像 x", "B × 3 × 224 × 224")
box(700, 118, 290, 78, "当前 task 类别 K", "Task t: K=5,8,…,26")
box(55, 236, 330, 92, "图像复制为 2K 条路径", "B → 2KB；每类负 / 正两条路径")
box(
    665, 231, 345, 97,
    "正 / 负 visual prompts",
    "每类两套 Vc− / Vc+\n16 × 768；共 2K 条路径",
    stroke: orange,
    fill: color(0xfffbeb),
    titleColor: color(0x9a3412)
)
box(
    250, 370, 570, 98,
    "冻结 CLIP ViT-B/16 Visual Encoder",
    "visual prompts 注入第 7–11 层 Q/K/V\n输出 B × 2K × 512 × 197",
    stroke: blue,
    fill: color(0xf8fbff),
    titleColor: color(0x1d4ed8)
)
box(
    285, 520, 500, 82,
    "归一化 token features  Z",
    "Z ∈ R^(B×2K×512×197)；CLS = Z[:,:,:,0]",
    stroke: color(0x94a3b8),
    fill: color(0xffffff)
)

orthogonalArrow([(345, 157), (700, 157)], color: blue)
orthogonalArrow([(200, 196), (200, 236)], color: blue)
orthogonalArrow([(845, 196), (845, 231)], color: blue)
orthogonalArrow([(220, 328), (220, 348), (390, 348), (390, 370)], color: blue)
orthogonalArrow([(840, 328), (840, 348), (680, 348), (680, 370)], color: blue)
orthogonalArrow([(535, 468), (535, 520)], color: blue)

// Text branch.
box(1245, 115, 365, 74, "EMOTIC 类别", "26 类；B5-C3 共 8 个 tasks")
box(
    1095, 230, 285, 84,
    "Positive prompt",
    "a photo of a person clearly feeling {class}",
    stroke: red,
    fill: color(0xfffbfb),
    titleColor: color(0xb91c1c),
    titleSize: 19,
    bodySize: 12
)
box(
    1473, 230, 285, 84,
    "Negative prompt",
    "a photo of a person not feeling {class}",
    stroke: red,
    fill: color(0xfffbfb),
    titleColor: color(0xb91c1c),
    titleSize: 19,
    bodySize: 12
)
box(
    1215, 350, 430, 86,
    "CSC Prompt Learner",
    "正 / 负 context 均为 7 tokens × 768\n每类独立可学习语义 context",
    stroke: orange,
    fill: color(0xfffbeb),
    titleColor: color(0x9a3412),
    bodySize: 14
)
box(
    1215, 470, 430, 76,
    "冻结 CLIP Text Encoder",
    "Transformer + text projection；normalize",
    stroke: red,
    fill: color(0xfffbfb),
    titleColor: color(0xb91c1c)
)
box(
    1215, 570, 430, 60,
    "文本特征 E− / E+",
    "2K × 512；按 seen classes 累积",
    stroke: color(0xb8c4d8),
    fill: color(0xffffff),
    titleSize: 19,
    bodySize: 13
)

orthogonalArrow([(1428, 189), (1428, 210), (1238, 210), (1238, 230)], color: red)
orthogonalArrow([(1428, 189), (1428, 210), (1615, 210), (1615, 230)], color: red)
orthogonalArrow([(1238, 314), (1238, 330), (1430, 330), (1430, 350)], color: red)
orthogonalArrow([(1615, 314), (1615, 330), (1430, 330), (1430, 350)], color: red)
orthogonalArrow([(1430, 436), (1430, 470)], color: red)
orthogonalArrow([(1430, 546), (1430, 570)], color: red)

// Internal Adapter row.
box(
    42, 738, 310, 118,
    "原始 DDP 路径 logits",
    "S = 20·einsum(Z,E)\nw = softmax(S⁺, token)；正 / 负路径共享\nL⁰ = 5·Σₙ(S ⊙ w),  B × 2K",
    stroke: blue,
    fill: color(0xf8fbff),
    titleColor: color(0x1d4ed8),
    titleSize: 19,
    bodySize: 13,
    titleFraction: 0.30
)
box(
    395, 748, 250, 98,
    "提取每条路径 CLS",
    "z = Z[:,:,:,0]\nB × 2K × 512",
    stroke: color(0xb8c4d8),
    fill: color(0xffffff),
    titleSize: 18,
    bodySize: 14
)
box(
    687, 728, 395, 138,
    "Shared Residual Adapter",
    fixedGlobalAlpha
        ? "Aα(z)=z+αW↑ GELU(W↓ norm(z))\n512 → 128 → 512；无 bias；131,072 参数\nα=0.03 对所有 task / class 固定\nW↓/W↑ 来自 Base5 16-shot；迁移后冻结"
        : "Aα(z)=z+αW↑ GELU(W↓ norm(z))\n512 → 128 → 512；无 bias；131,072 参数\nW↓/W↑ 来自 Base5 16-shot Prototype Adapter\n3 seeds；迁移后冻结，DDP 内部不重训",
    stroke: orange,
    fill: color(0xfffbeb),
    titleColor: color(0x9a3412),
    titleSize: 20,
    bodySize: 13,
    titleFraction: 0.25
)
box(
    1125, 746, 305, 102,
    fixedGlobalAlpha ? "CLS residual → logit correction" : "CLS logit correction",
    fixedGlobalAlpha
        ? "Δz = Aα(z)−z\nΔL = 100·〈Δz, E〉；Lfinal = LDDP + ΔL"
        : "ΔLα = 100·〈Aα(z)−z, E〉\nLα = L⁰ + ΔLα",
    stroke: orange,
    fill: color(0xfffbeb),
    titleColor: color(0x9a3412),
    titleSize: 18,
    bodySize: 14
)
box(
    1480, 742, 270, 110,
    fixedGlobalAlpha ? "最终 task scores" : "两路 task scores",
    fixedGlobalAlpha
        ? "p = softmax(Lfinal/T)_pos\n所有 seen classes 使用同一分支\nT 沿用原 DDP task schedule"
        : "p⁰ = softmax(L⁰/T)_pos\npα = softmax(Lα/T)_pos\nT 使用原 DDP task schedule",
    stroke: purple,
    fill: color(0xf5f3ff),
    titleColor: color(0x6d28d9),
    titleSize: 18,
    bodySize: 13
)

// Cross-panel inputs use whitespace lanes and never enter another module.
orthogonalArrow([(500, 602), (500, 704), (195, 704), (195, 738)], color: blue)
orthogonalArrow([(570, 602), (570, 718), (520, 718), (520, 748)], color: blue)
orthogonalArrow([(1430, 630), (1430, 690), (1278, 690), (1278, 746)], color: red)

orthogonalArrow([(645, 797), (687, 797)], color: green)
orthogonalArrow([(1082, 797), (1125, 797)], color: orange)
orthogonalArrow([(1430, 797), (1480, 797)], color: purple)

if fixedGlobalAlpha {
    // L_DDP reaches the add box through a dedicated lower lane; it never crosses
    // the CLS or text connectors.
    orthogonalArrow([(197, 856), (197, 888), (1278, 888), (1278, 848)], color: blue)

    box(
        52, 938, 320, 112,
        "固定全局超参数 α=0.03",
        "Task 0 validation 一次选定并锁定\n所有 task、所有 class 共用；不再逐类挑选",
        stroke: purple,
        fill: color(0xf5f3ff),
        titleColor: color(0x6d28d9),
        titleSize: 18,
        bodySize: 13
    )
    box(
        414, 938, 345, 112,
        "Adapter 权重来源（离线）",
        "仅用 Base5 train 的 16-shot / class\n3 seeds；迁移 W↓/W↑；不使用 val/test 训练权重",
        stroke: orange,
        fill: color(0xfffbeb),
        titleColor: color(0x9a3412),
        titleSize: 18,
        bodySize: 13
    )
    box(
        802, 938, 320, 112,
        "统一增量推理",
        "Lfinal 对全部 seen classes 直接预测\n无 class gate；无 βc；无外部模型分数融合",
        stroke: green,
        fill: color(0xffffff),
        titleColor: color(0x047857),
        titleSize: 19,
        bodySize: 13
    )
    box(
        1164, 938, 245, 112,
        "严格评估",
        "val 阈值仅用于 cF1 / oF1\nmAP 直接使用连续 scores\ntest 不参与 α 或权重选择",
        stroke: green,
        fill: color(0xf0fdf4),
        titleColor: color(0x047857),
        titleSize: 18,
        bodySize: 12
    )
    box(
        1451, 928, 300, 132,
        "Fixed-global-α 结果",
        "Final Test mAP\n30.8191 → 31.3129 ± 0.2036\n平均增益 +0.4938；3 seeds\nAverage task mAP: 38.5126 ± 0.2543",
        stroke: purple,
        fill: color(0xf5f3ff),
        titleColor: color(0x6d28d9),
        titleSize: 18,
        bodySize: 12,
        titleFraction: 0.23
    )

    orthogonalArrow([(372, 994), (414, 994)], color: orange)
    orthogonalArrow([(759, 994), (802, 994)], color: green)
    orthogonalArrow([(1122, 994), (1164, 994)], color: green)
    orthogonalArrow([(1409, 994), (1451, 994)], color: purple)
} else {
    orthogonalArrow([(197, 856), (197, 890), (1615, 890), (1615, 852)], color: blue)

    // Strict val selection and held-out test row.
    box(
        52, 944, 320, 102,
        "val：选择 task-level αₜ",
        "α ∈ {0,.001,.003,.01,.03}\n仅当 task val mAP 增益 > 0.1 才启用",
        stroke: purple,
        fill: color(0xf5f3ff),
        titleColor: color(0x6d28d9),
        titleSize: 18,
        bodySize: 13
    )
    box(
        414, 936, 345, 118,
        "val：逐类二值 class gate",
        "gₜ,c = 1  iff  APval(pα,c)−APval(p⁰,c)>0.1\n否则 gₜ,c = 0；不是每类学习连续 β",
        stroke: purple,
        fill: color(0xf5f3ff),
        titleColor: color(0x6d28d9),
        titleSize: 18,
        bodySize: 13,
        titleFraction: 0.28
    )
    box(
        802, 944, 320, 102,
        "冻结增量推理",
        "p_final,c = gₜ,c pα,c + (1−gₜ,c)p⁰_c\n每个 task 只对 seen classes 预测",
        stroke: green,
        fill: color(0xffffff),
        titleColor: color(0x047857),
        titleSize: 19,
        bodySize: 13
    )
    box(
        1164, 944, 245, 102,
        "val 阈值 → test",
        "阈值只为 cF1 / oF1\nmAP 直接使用连续 scores",
        stroke: green,
        fill: color(0xf0fdf4),
        titleColor: color(0x047857),
        titleSize: 18,
        bodySize: 13
    )
    box(
        1451, 930, 300, 128,
        "CLS class-gate 结果",
        "Final Test mAP\n30.8051 → 31.8938 ± 0.1041\n增益 +1.0887；3 seeds\nAverage mAP: 38.8350 ± 0.1636",
        stroke: purple,
        fill: color(0xf5f3ff),
        titleColor: color(0x6d28d9),
        titleSize: 19,
        bodySize: 13,
        titleFraction: 0.23
    )

    orthogonalArrow([(1615, 852), (1615, 915), (212, 915), (212, 944)], color: purple)
    orthogonalArrow([(372, 995), (414, 995)], color: purple)
    orthogonalArrow([(759, 995), (802, 995)], color: green)
    orthogonalArrow([(1122, 995), (1164, 995)], color: green)
    orthogonalArrow([(1409, 995), (1451, 995)], color: purple)
}

// Legend and protocol note.
let strip = rectTop(14, 1106, 1771, 78)
let dashed = NSBezierPath(roundedRect: strip, xRadius: 8, yRadius: 8)
dashed.setLineDash([8, 6], count: 2, phase: 0)
color(0xffffff, 0.78).setFill()
dashed.fill()
color(0x94a3b8).setStroke()
dashed.lineWidth = 1.6
dashed.stroke()

legendItem(42, 1134, orange, color(0xfffbeb), "Adapter 权重：Base5 16-shot 训练后迁移并冻结", width: 380)
legendItem(
    510, 1134, blue, color(0xf8fbff),
    fixedGlobalAlpha ? "冻结：DDP task checkpoint + CLIP encoders" : "冻结：CLIP image/text encoders",
    width: 300
)
legendItem(
    875, 1134, purple, color(0xf5f3ff),
    fixedGlobalAlpha ? "全局 α=0.03：一次选择后固定，不设逐类 gate" : "仅 val 选择：αₜ / class gate / threshold",
    width: 360
)
drawText(
    fixedGlobalAlpha
        ? "严格协议：Base5 train 只训练 Adapter 权重；Task 0 val 仅选一个全局 α；test 不参与调参。推理无第二个 CLIP、无外部分数融合。"
        : "严格协议：train 只训练权重来源；val 做选择；test 不参与调参。推理无第二个 CLIP、无外部分数融合。",
    in: rectTop(1270, 1120, 485, 48),
    size: 14,
    weight: .semibold,
    color: slate,
    align: .left
)
}

image.unlockFocus()

guard let tiff = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiff),
      let data = bitmap.representation(using: .png, properties: [:]) else {
    fatalError("Failed to create PNG data")
}

try data.write(to: URL(fileURLWithPath: outputPath))
print(outputPath)
