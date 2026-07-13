import AppKit

let W: CGFloat = 1800
let H: CGFloat = 1200

let outputPath = "docs/figures/code_ddp_external_fusion_adapter_architecture.png"

func color(_ hex: UInt32, _ alpha: CGFloat = 1.0) -> NSColor {
    let r = CGFloat((hex >> 16) & 0xff) / 255.0
    let g = CGFloat((hex >> 8) & 0xff) / 255.0
    let b = CGFloat(hex & 0xff) / 255.0
    return NSColor(calibratedRed: r, green: g, blue: b, alpha: alpha)
}

func rectTop(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat) -> NSRect {
    return NSRect(x: x, y: H - y - h, width: w, height: h)
}

func pointTop(_ x: CGFloat, _ y: CGFloat) -> NSPoint {
    return NSPoint(x: x, y: H - y)
}

func drawText(
    _ text: String,
    in rect: NSRect,
    size: CGFloat = 22,
    weight: NSFont.Weight = .regular,
    color textColor: NSColor = .black,
    align: NSTextAlignment = .center
) {
    let para = NSMutableParagraphStyle()
    para.alignment = align
    para.lineBreakMode = .byWordWrapping
    para.lineSpacing = 2
    let font = NSFont.systemFont(ofSize: size, weight: weight)
    let attrs: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: textColor,
        .paragraphStyle: para
    ]
    NSAttributedString(string: text, attributes: attrs).draw(in: rect.insetBy(dx: 8, dy: 2))
}

func fillRoundRect(_ r: NSRect, radius: CGFloat, fill: NSColor, stroke: NSColor, lineWidth: CGFloat = 2) {
    let path = NSBezierPath(roundedRect: r, xRadius: radius, yRadius: radius)
    fill.setFill()
    path.fill()
    stroke.setStroke()
    path.lineWidth = lineWidth
    path.stroke()
}

func box(
    _ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat,
    _ title: String,
    _ body: String = "",
    stroke: NSColor = color(0xb8c4d8),
    fill: NSColor = color(0xffffff, 0.92),
    titleColor: NSColor = .black,
    titleSize: CGFloat = 22,
    bodySize: CGFloat = 16,
    radius: CGFloat = 8
) {
    let r = rectTop(x, y, w, h)
    fillRoundRect(r, radius: radius, fill: fill, stroke: stroke, lineWidth: 1.8)
    let titleRect = NSRect(x: r.minX + 4, y: r.minY + h * 0.48, width: r.width - 8, height: h * 0.45)
    drawText(title, in: titleRect, size: titleSize, weight: .semibold, color: titleColor)
    if !body.isEmpty {
        let bodyRect = NSRect(x: r.minX + 4, y: r.minY + 6, width: r.width - 8, height: h * 0.54)
        drawText(body, in: bodyRect, size: bodySize, weight: .regular, color: color(0x111827))
    }
}

func panel(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat, fill: NSColor, stroke: NSColor, title: String, titleColor: NSColor) {
    let r = rectTop(x, y, w, h)
    fillRoundRect(r, radius: 10, fill: fill, stroke: stroke, lineWidth: 2.2)
    drawText(title, in: rectTop(x + 16, y + 8, w - 32, 34), size: 24, weight: .bold, color: titleColor, align: .left)
}

func arrow(_ sx: CGFloat, _ sy: CGFloat, _ ex: CGFloat, _ ey: CGFloat, color c: NSColor, width: CGFloat = 4) {
    let start = pointTop(sx, sy)
    let end = pointTop(ex, ey)
    let path = NSBezierPath()
    path.move(to: start)
    path.line(to: end)
    c.setStroke()
    path.lineWidth = width
    path.lineCapStyle = .round
    path.stroke()

    let angle = atan2(end.y - start.y, end.x - start.x)
    let len: CGFloat = 16
    let spread: CGFloat = .pi / 7
    let p1 = NSPoint(x: end.x - len * cos(angle - spread), y: end.y - len * sin(angle - spread))
    let p2 = NSPoint(x: end.x - len * cos(angle + spread), y: end.y - len * sin(angle + spread))
    let head = NSBezierPath()
    head.move(to: end)
    head.line(to: p1)
    head.line(to: p2)
    head.close()
    c.setFill()
    head.fill()
}

func elbowArrow(_ points: [(CGFloat, CGFloat)], color c: NSColor, width: CGFloat = 4) {
    guard points.count >= 2 else { return }
    for index in 1..<points.count {
        let a = points[index - 1]
        let b = points[index]
        precondition(a.0 == b.0 || a.1 == b.1, "elbowArrow segments must be horizontal or vertical")
    }
    let path = NSBezierPath()
    path.move(to: pointTop(points[0].0, points[0].1))
    for p in points.dropFirst() {
        path.line(to: pointTop(p.0, p.1))
    }
    c.setStroke()
    path.lineWidth = width
    path.lineCapStyle = .round
    path.lineJoinStyle = .round
    path.stroke()

    let a = points[points.count - 2]
    let b = points[points.count - 1]
    let start = pointTop(a.0, a.1)
    let end = pointTop(b.0, b.1)
    let angle = atan2(end.y - start.y, end.x - start.x)
    let len: CGFloat = 16
    let spread: CGFloat = .pi / 7
    let p1 = NSPoint(x: end.x - len * cos(angle - spread), y: end.y - len * sin(angle - spread))
    let p2 = NSPoint(x: end.x - len * cos(angle + spread), y: end.y - len * sin(angle + spread))
    let head = NSBezierPath()
    head.move(to: end)
    head.line(to: p1)
    head.line(to: p2)
    head.close()
    c.setFill()
    head.fill()
}

func legendItem(_ x: CGFloat, _ y: CGFloat, _ stroke: NSColor, _ fill: NSColor, _ label: String) {
    fillRoundRect(rectTop(x, y, 28, 24), radius: 5, fill: fill, stroke: stroke, lineWidth: 2)
    drawText(label, in: rectTop(x + 38, y - 2, 360, 30), size: 16, weight: .regular, color: color(0x111827), align: .left)
}

let image = NSImage(size: NSSize(width: W, height: H))
image.lockFocus()

// Background
color(0xf8fbff).setFill()
NSRect(x: 0, y: 0, width: W, height: H).fill()

// Title
drawText(
    "CODE_DDP: EMOTIC B5-C3 External Prototype-Adapter Fusion",
    in: rectTop(0, 0, W, 46),
    size: 34,
    weight: .bold,
    color: color(0x0f172a)
)

let blue = color(0x2563eb)
let red = color(0xdc2626)
let green = color(0x059669)
let orange = color(0xf97316)
let purple = color(0x7c3aed)
let slate = color(0x475569)

// Panels
panel(14, 56, 850, 600, fill: color(0xeef6ff, 0.92), stroke: blue, title: "DDP 主分支：class-specific dual visual prompts", titleColor: color(0x1d4ed8))
panel(890, 56, 895, 600, fill: color(0xfff1f2, 0.92), stroke: red, title: "外部 Prototype-Adapter 分支：frozen CLIP global feature", titleColor: color(0xb91c1c))
panel(14, 682, 1771, 390, fill: color(0xecfdf5, 0.95), stroke: green, title: "", titleColor: color(0x047857))
drawText("外部融合器：val 选择，test 评估，不重新训练 DDP", in: rectTop(560, 690, 680, 34), size: 24, weight: .bold, color: color(0x047857))

// DDP branch boxes
box(60, 115, 290, 76, "输入图像 x", "B × 3 × 224 × 224", stroke: color(0xb8c4d8))
box(510, 115, 290, 76, "当前 task 类别 K", "Task t: K=5,8,...,26", stroke: color(0xb8c4d8))
box(60, 236, 310, 86, "图像复制为 2K 条路径", "B → 2KB  正/负 × 类别", stroke: color(0xb8c4d8))
box(490, 226, 330, 96, "正/负 visual prompts", "每类两套 Vc− / Vc+\n16 × 768；2K 路径", stroke: orange, fill: color(0xfffbeb), titleColor: color(0x9a3412))
box(145, 365, 590, 92, "冻结 CLIP ViT-B/16 Visual Encoder", "visual prompts 注入第 7–11 层 Q/K/V\n输出 token features：B × 2K × 512 × 197", stroke: blue, fill: color(0xf8fbff), titleColor: color(0x111827))
box(70, 485, 300, 72, "token-wise similarity", "S = 20 · einsum(Z, E)", stroke: color(0xb8c4d8))
box(480, 485, 300, 72, "负类 token attention", "w = softmax(Sneg token)\n正/负共享 w", stroke: color(0xb8c4d8))
box(280, 580, 315, 66, "DDP task scores", "p_DDP / logits: B × K；由 task*_scores.pt 读入", stroke: blue, fill: color(0xf8fbff), titleColor: color(0x1d4ed8), titleSize: 19, bodySize: 13)

// DDP branch orthogonal connectors
arrow(350, 153, 510, 153, color: blue)
arrow(205, 191, 205, 236, color: blue)
arrow(655, 191, 655, 226, color: blue)
elbowArrow([(215, 322), (215, 344), (305, 344), (305, 365)], color: blue)
elbowArrow([(655, 322), (655, 344), (575, 344), (575, 365)], color: blue)
elbowArrow([(440, 457), (440, 474), (220, 474), (220, 485)], color: green)
elbowArrow([(440, 457), (440, 474), (630, 474), (630, 485)], color: green)
elbowArrow([(220, 557), (220, 568), (380, 568), (380, 580)], color: green)
elbowArrow([(630, 557), (630, 568), (495, 568), (495, 580)], color: green)

// Prototype branch boxes
box(930, 115, 300, 76, "输入图像 x", "同一张 full image", stroke: color(0xb8c4d8))
box(1425, 110, 320, 86, "固定正/负文本原型", "t_c+ / t_c−\n3 个模板平均；不随 DDP 学习", stroke: red, fill: color(0xfffbfb), titleColor: color(0xb91c1c))
box(930, 236, 350, 96, "一次普通 CLIP 图像编码", "冻结 ViT-B/16；无 visual prompt\nz = Enc_img(I),  B × 512", stroke: blue, fill: color(0xf8fbff), titleColor: color(0x1d4ed8))
box(1395, 236, 350, 96, "冻结 CLIP Text Encoder", "positive / negative templates\n得到 26 × 512 原型", stroke: red, fill: color(0xfffbfb), titleColor: color(0xb91c1c))
box(930, 380, 400, 106, "Shared Residual Adapter", "z′ = norm(z + γ W₂ GELU(W₁z))\n512 → 128 → 512, γ=0.1, W₂ zero-init", stroke: orange, fill: color(0xfffbeb), titleColor: color(0x9a3412), bodySize: 15)
box(1395, 380, 350, 106, "Prototype logits", "l_c = s[cos(z′,t_c+) − cos(z′,t_c−)]\n再做 val calibration", stroke: red, fill: color(0xfffbfb), titleColor: color(0xb91c1c), bodySize: 14)
box(1400, 560, 345, 80, "Prototype scores", "p_proto = sigmoid(calibrated logits)\nB × K，按 seen classes 截取", stroke: purple, fill: color(0xf5f3ff), titleColor: color(0x6d28d9), titleSize: 20, bodySize: 14)

// Prototype branch orthogonal connectors
elbowArrow([(1080, 191), (1080, 214), (1105, 214), (1105, 236)], color: blue)
elbowArrow([(1585, 196), (1585, 214), (1570, 214), (1570, 236)], color: red)
elbowArrow([(1105, 332), (1105, 356), (1130, 356), (1130, 380)], color: orange)
arrow(1570, 332, 1570, 380, color: red)
arrow(1330, 433, 1395, 433, color: red)
arrow(1570, 486, 1570, 560, color: purple)

// Fusion panel boxes
box(55, 755, 260, 90, "DDP 分数", "p_DDP：来自已训练 DDP\n不改 forward，不重训", stroke: blue, fill: color(0xf8fbff), titleColor: color(0x1d4ed8))
box(1500, 755, 250, 90, "Prototype 分数", "p_proto：zero/few/full base5\n外部分支离线产生", stroke: purple, fill: color(0xf5f3ff), titleColor: color(0x6d28d9), bodySize: 14)
box(1170, 750, 280, 100, "val 上校准 Prototype", "temperature + bias\n只用 val labels；test 不参与选择", stroke: orange, fill: color(0xfffbeb), titleColor: color(0x9a3412), bodySize: 14)
box(400, 750, 330, 100, "val 上选择融合策略", "global β sweep: 0...1\nbinary class gate: β_c ∈ {0, β}", stroke: green, fill: color(0xf0fdf4), titleColor: color(0x047857), bodySize: 14)
box(650, 915, 360, 100, "Score-level Fusion", "global: p = (1−β)p_DDP + βp_proto\nclass gate: p_c = (1−β_c)p_DDP,c + β_c p_proto,c", stroke: green, fill: color(0xffffff), titleColor: color(0x047857), bodySize: 14)
box(1060, 918, 200, 95, "阈值选择", "只在 val 上选\n用于 F1", stroke: green, fill: color(0xf0fdf4), titleColor: color(0x047857), titleSize: 20, bodySize: 14)
box(1300, 918, 200, 95, "test 评估", "mAP 用 scores\ncF1 / oF1 用 val 阈值", stroke: color(0xb8c4d8), fill: color(0xffffff), titleColor: color(0x111827), titleSize: 20, bodySize: 13)
box(1540, 918, 200, 95, "输出", "JSON/PT + per-class AP\n30.819 → 33.845", stroke: purple, fill: color(0xf5f3ff), titleColor: color(0x6d28d9), titleSize: 20, bodySize: 13)

// Cross-panel routes enter from the top through dedicated whitespace channels.
elbowArrow([(437, 646), (437, 735), (185, 735), (185, 755)], color: blue)
elbowArrow([(1570, 640), (1570, 735), (1625, 735), (1625, 755)], color: purple)

// Fusion logic: scores -> calibration / selector -> fixed score-level fusion.
arrow(315, 800, 400, 800, color: blue)
arrow(1500, 800, 1450, 800, color: purple)
arrow(1170, 800, 730, 800, color: orange)
elbowArrow([(315, 825), (350, 825), (350, 965), (650, 965)], color: blue)
elbowArrow([(565, 850), (565, 880), (830, 880), (830, 915)], color: green)
elbowArrow([(1310, 850), (1310, 885), (950, 885), (950, 915)], color: orange)
arrow(1010, 965, 1060, 965, color: green)
arrow(1260, 965, 1300, 965, color: green)
arrow(1500, 965, 1540, 965, color: purple)

// Notes strip
let strip = rectTop(14, 1090, 1771, 92)
let dash = NSBezierPath(roundedRect: strip, xRadius: 8, yRadius: 8)
dash.setLineDash([8, 6], count: 2, phase: 0)
color(0xffffff, 0.75).setFill()
dash.fill()
color(0x94a3b8).setStroke()
dash.lineWidth = 1.6
dash.stroke()
legendItem(46, 1124, orange, color(0xfffbeb), "可学习：Prototype Adapter 或 DDP prompts")
legendItem(520, 1124, blue, color(0xf8fbff), "冻结：CLIP image/text encoders")
legendItem(920, 1124, purple, color(0xf5f3ff), "外部融合：读取特征缓存与 task scores")
drawText("严格设置：Base5/Few-shot 只用训练集 base 类标签；后续 task 冻结迁移；All26 只作为上限实验", in: rectTop(1330, 1107, 420, 62), size: 15, weight: .semibold, color: color(0x334155), align: .left)

image.unlockFocus()

guard let tiff = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiff),
      let data = bitmap.representation(using: .png, properties: [:]) else {
    fatalError("Failed to create PNG data")
}

try data.write(to: URL(fileURLWithPath: outputPath))
print(outputPath)
