# EMSlite Dashboard — Developer Implementation Prompt

## What You Are Building

Build the **EMSlite Energy Monitoring Dashboard** — an admin-style single-page application with a fixed sidebar, sticky topbar, and a scrollable main content area. The dashboard visualises energy consumption, carbon reduction, generation output, and cost savings across US states.

---

## Layout Shell

The page is divided into three persistent regions:

1. **Sidebar (left):** A full-height dark panel containing a brand logo at the top, a vertical navigation list in the middle, and a logout action pinned to the bottom. The sidebar can collapse to icon-only mode on smaller viewports and expand to show icon+label on larger ones.

2. **Topbar (top, right of sidebar):** A slim horizontal bar containing — on the right side — a language selector (flag icon + label + dropdown arrow), a notification bell icon, and a circular user avatar with a dropdown arrow. The left side is empty or reserved for breadcrumbs.

3. **Main Content (below topbar, right of sidebar):** A padded, scrollable area with a light tinted background that holds all dashboard widgets.

---

## Main Content Sections (top to bottom)

### Row 1 — KPI Summary Cards (3 cards, equal width)

Three cards displayed in a horizontal row:

| Card | Title | Hero Value | Badge |
|------|-------|-----------|-------|
| 1 | Total Carbon Reduction | 73t | ▲ 12.97% |
| 2 | Generation | 215,440 | ▲ 12.97% |
| 3 | Savings Made | $74,013 | ▲ 12.97% |

Each card contains:
- A **title** in small muted text at the top-left.
- A **hero value** in large bold text below the title.
- A **trend badge** (pill shape) showing a percentage with an upward arrow icon. Green tint for positive trends.
- A **subtitle** below the badge reading "Compared Last Month" in very small muted text.
- A **decorative icon** in the top-right corner — a circular light-green background with a themed glyph (leaf for carbon, bolt for generation, coin for savings).

Cards sit on a white surface with subtle shadow and generous rounded corners.

### Row 2 — Charts Grid

Below the KPI row, four chart panels occupy a 12-column grid:

#### Panel A — Consumption by State (hex tile map) · ~5 columns
A hexagonal tile map of US states. Each hexagon displays a 2-letter state abbreviation and is color-coded by consumption level:
- **Low:** bright green (primary accent)
- **Medium:** orange
- **High:** coral/red
- **Very High / baseline:** dark navy

A pair of zoom +/– buttons sit in the bottom-left corner of the panel.

#### Panel B — Energy Use (vertical bar chart) · below or combined with Panel A
A bar chart with approximately 12 vertical bars. Each bar uses a bottom-to-top gradient transitioning from deep navy to bright green. Bars have slightly rounded top caps. Category labels run along the x-axis.

#### Panel C — Energy Utilisation Efficiency (donut chart) · ~3 columns
A two-segment donut ring:
- **Solar Energy — 43%** in bright green.
- **Off-Peak Energy — 57%** in dark navy.

A legend below the donut shows colored dots with labels and percentages.

#### Panel D — States for Energy Consumption (horizontal bar chart) · ~4 columns
A ranked list of 5 states (Virginia, Mexico, Nevada, New York, Canada) each showing:
- State name on the left.
- A horizontal progress-style bar (dark navy fill on a light track, fully rounded pill shape).
- kWh value on the right.

---

## Token Rules & Usage

All styling must reference the **design token system** defined in `ui-design-brief.jsonc`. Here is how tokens map to usage:

### Color Tokens

| Token | Where to use |
|-------|-------------|
| `primary-400` | Sidebar active nav background, positive trend badge icon color, donut solar segment, hex-map low-consumption fill, bar chart gradient top stop, decorative KPI icon backgrounds |
| `primary-50 … primary-200` | Light tinted backgrounds behind decorative KPI icons, hover highlights on primary buttons |
| `secondary-800 … secondary-950` | Sidebar background, horizontal bar fills, donut off-peak segment, bar chart gradient bottom stop, hex-map high-consumption fill |
| `gray-50` | Page background (light mode) |
| `gray-200` | Card borders, dividers, input borders, bar chart track (light mode) |
| `gray-400 / gray-500` | Muted text, subtitles, axis labels |
| `gray-600 / gray-700` | Primary body text / headings (light mode) |
| `accents.positive` | Trend badge background + text (green pill) |
| `accents.negative` | Inverse trend badge (red pill) — wire up even if not visible in screenshot |
| `accents.warning` | Hex-map medium-consumption tiles |
| `accents.danger` | Hex-map high-consumption tiles |
| `accents.chartGradient` | Vertical bar chart fill gradient |

### Surface Token

Every card, panel, and popover uses the `tokens.surface` composite:
- **Light:** white background, `gray-200` border, soft shadow.
- **Dark:** `gray-900` background, `gray-800` border, deeper shadow.

### Badge Tokens

Use `tokens.badge-positive` for upward trends and `tokens.badge-negative` for downward trends. In dark mode swap to the dark variants which use translucent backgrounds and lighter text for readability.

### Nav Tokens

- Active item: `tokens.nav-item-active` — primary-400 background, dark text, semibold, rounded corners.
- Inactive item: `tokens.nav-item-inactive` — white text at 70% opacity, translucent white hover background.

---

## Dark Mode

Support class-based dark mode toggling. The sidebar already uses dark colors and does not change. All other surfaces, text, borders, and shadows swap per the `darkMode.mappings` table in the JSONC spec. Key rules:

- Page background shifts from near-white to near-black.
- Cards shift from white to dark gray.
- Text inverts from dark-on-light to light-on-dark.
- Positive/negative badges switch from solid tinted backgrounds to translucent overlays to avoid harsh contrast.
- Chart colors remain largely the same (they already use strong brand colors); only track/grid colors lighten or darken for legibility.

---

## Responsive Behavior

Follow standard breakpoints (640 / 768 / 1024 / 1280 / 1536 px):

| Breakpoint | Behavior |
|-----------|----------|
| < 640px (mobile) | Sidebar hidden (hamburger toggle). Topbar full-width. KPI cards stack vertically. All chart panels stack full-width. |
| 640–767px | Sidebar collapses to icon-only rail. KPI cards still single column. Charts still stacked. |
| 768–1023px | KPI cards in 2-column grid (third wraps). Charts in 2-column layout (map+bars left, donut+state-bars right). |
| 1024px+ | Sidebar fully expanded. KPI cards 3-column. Charts adopt designed grid spans. |
| 1280px+ | Comfortable spacing, no structural changes. |
| 1536px+ | Optional max-width container to prevent over-stretching on ultrawide displays. |

---

## General Notes

- All border radii on cards and panels should be generous (1rem / 16px).
- Badge pills are fully rounded (9999px).
- Spacing between cards, panels, and sections is consistent (1.5rem / 24px gap).
- The sidebar active indicator has a subtle rounded rectangle background, not a left-edge stripe.
- Chart tooltips should follow the surface token (card-like popover) and appear on hover.
- The notification bell should support an optional unread indicator dot (small primary-colored circle, absolute-positioned top-right of the icon).
- The language switcher displays a small country flag image; support swapping flag + label for locale changes.
