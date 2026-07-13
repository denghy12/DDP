import AppKit

let W: CGFloat = 1800
let H: CGFloat = 1200

let outputPath = "docs/figures/code_ddp_internal_cls_class_gate_adapter_architecture.png"

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

let image = NSImage(size: NSSize(width: W, height: H))
image.lockFocus()

color(0xf8fbff).setFill()
NSRect(x: 0, y: 0, width: W, height: H).fill()

drawText(
    "CODE_DDP: EMOTIC B5-C3 Internal CLS Adapter + Task Alpha + Class Gate",
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
    title: "DDP 内部共享 CLS Adapter：一次 DDP 编码；val 选择；test 只评估",
    titleColor: color(0x047857),
    centered: true
)

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
    "Aα(z)=z+αW↑ GELU(W↓ norm(z))\n512 → 128 → 512；无 bias；131,072 参数\nW↓/W↑ 来自 Base5 16-shot Prototype Adapter\n3 seeds；迁移后冻结，DDP 内部不重训",
    stroke: orange,
    fill: color(0xfffbeb),
    titleColor: color(0x9a3412),
    titleSize: 20,
    bodySize: 13,
    titleFraction: 0.25
)
box(
    1125, 746, 305, 102,
    "CLS logit correction",
    "ΔLα = 100·〈Aα(z)−z, E〉\nLα = L⁰ + ΔLα",
    stroke: orange,
    fill: color(0xfffbeb),
    titleColor: color(0x9a3412),
    titleSize: 18,
    bodySize: 14
)
box(
    1480, 742, 270, 110,
    "两路 task scores",
    "p⁰ = softmax(L⁰/T)_pos\npα = softmax(Lα/T)_pos\nT 使用原 DDP task schedule",
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
legendItem(510, 1134, blue, color(0xf8fbff), "冻结：CLIP image/text encoders", width: 300)
legendItem(875, 1134, purple, color(0xf5f3ff), "仅 val 选择：αₜ / class gate / threshold", width: 360)
drawText(
    "严格协议：train 只训练权重来源；val 做选择；test 不参与调参。推理无第二个 CLIP、无外部分数融合。",
    in: rectTop(1270, 1120, 485, 48),
    size: 14,
    weight: .semibold,
    color: slate,
    align: .left
)

image.unlockFocus()

guard let tiff = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiff),
      let data = bitmap.representation(using: .png, properties: [:]) else {
    fatalError("Failed to create PNG data")
}

try data.write(to: URL(fileURLWithPath: outputPath))
print(outputPath)
