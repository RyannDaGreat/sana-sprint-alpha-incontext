/**
 * Full-resolution eval grids for the journal. One row per eval item, one cell per column, every cell the stored
 * 1024x1024 PNG (hard link under assets/eval/full/, made by src/sana_alpha/journal_assets.py). Click a cell to open
 * the file. The browser scales cells to fit the window; "native size" shows them 1:1 with a horizontal scroll. Under each model cell: its weight equation, MODEL_LABELS from eval_labels.js (load it first).
 *
 * @example evalGrid("#grid", [["matte", "matte"], ["sprint_gray", "stock Sprint"]])
 */
const EVAL_FULL_DIR = "assets/eval/full/";
const EVAL_ITEMS = [  // [page_id, caption], in the order of resources/eval/prompts.json
  ["1qda4tXFQy", "portrait of a bored yeoman woman stubble running"],
  ["1wgVB7AefT", "two red hearts shape on stick over . valentine concept. selective focus."],
  ["565tDciDuD", "cork and wire ."],
  ["6NqAw3m1ss", "attractive young girl holds in her hands two red dice with serious gesture in front of an intense"],
  ["95f0C6wQyn", "A transgender man with a stylishly messy brunette hairdo. He sports a red bandana and a plain white tee. His broad grin revealing a playful personality, eyes framed by a hint of guyliner."],
  ["98nn0RzTU4", "upset 70 year old Indian baby postal worker man"],
  ["9PYbM7gvWC", "nonbinary swordsmith girl with strawberry blonde long hair matte red lipstick blue eyes sunglasses blue eyes sunglasses"],
  ["BTnRzkWMNh", "fresh organic mandarins on a turquoise"],
  ["Fqo7u0XZyx", "a man hand and gestures in black rubber glove shows knife ."],
  ["GLzmqMEwUO", "wide view of a teenage skinny baseball player girl hazel eyes freckles glossy dark lipstick"],
  ["NnfzyZv7cP", "Middle Eastern gay gay sorcerer man with white long flowing hair hazel eyes wearing brown heavy victorian dress coat"],
  ["RYEZcIFaZi", "white flower tulip  with clipping path. close-up. shot of white colored. nature."],
  ["RchOPcbQLG", "construction worker man with white flowing hair matte lipgloss beard pearl earrings"],
  ["SI6Izcw6Jy", "baked red apples with cinnamon, walnuts and honey . walnuts honey is a healthy dietary food."],
  ["V1YCu9hwmo", "red apple on a solid"],
  ["WWqakkJo0r", "vaccination of a covid virus cell 3d illustration,  two syringes and a covid cell"],
  ["Z4ncYHThJe", "a smiling young girl with trunk,"],
  ["b0CewcDGkW", "chef hat and knife outline logo.kitchen simple black icon. isolate ."],
  ["eSocyAFw7c", "close-up silvery and vintage spoon with honey falling,"],
  ["gcp6vpZDjq", "close up of a middle age Hispanic monster hunter woman big ears"],
  ["ijseKlXgsh", "green tea matcha latte in a glass cup . concept of a healthy diet, superfood, antioxidant, cleansing"],
  ["jSRxTkSuC1", "stone black and"],
  ["mQsPepMSfu", "portrait of a young geologist girl with spiky hair big ears"],
  ["mZaMuBqsEq", "funny handsome young man with dark suit with a beautiful bouquet in the hands of white and purple flowers"],
  ["sCPZQgnUPU", "big ham sandwich with lettuce, tomato and cheese ."],
  ["tMWV86u1Et", "b miejski, czarny ptak, kostka brukowa, rozmyte szare tåo"],
  ["uP1cKAy8ej", "tin can"],
  ["wtAc9ZTbJZ", "es blewah or contain slices of cantaloupe with grass jelly and syrup. served in bowl ."],
  ["zxrxYWORPC", "flowers petels and fruit twigs in spring on pastel ,flat lay"]
];

/**
 * Command. Fills the element with the grid.
 *
 * @param {string} selector - the container
 * @param {Array<[string, string]>} columns - [folder under assets/eval/full/, label under the cell]
 */
function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
}

function evalGrid(selector, columns) {
  const root = document.querySelector(selector);
  root.classList.add("evalgrid");
  root.style.setProperty("--cols", columns.length);
  const toggle = document.createElement("button");
  toggle.className = "gridtoggle";
  toggle.textContent = "native size (1:1)";
  toggle.onclick = () => toggle.setAttribute("aria-pressed", root.classList.toggle("native"));
  root.before(toggle);
  for (const [id, caption] of EVAL_ITEMS) {
    const row = document.createElement("div");
    row.className = "evalrow";
    row.innerHTML = columns.map(([dir, label]) =>
      `<figure><a href="${EVAL_FULL_DIR}${dir}/${id}.webp" target="_blank"><img loading="lazy" decoding="async" ` +
      `src="${EVAL_FULL_DIR}${dir}/${id}.webp" alt="${label}: ${escapeHtml(caption)}"></a><figcaption>${label}${dir in MODEL_LABELS ? `<br><small>${escapeHtml(MODEL_LABELS[dir])}</small>` : ""}</figcaption></figure>`).join("") +
      `<p class="rowcap"><b>${escapeHtml(caption)}</b> <code>${id}</code></p>`;
    root.appendChild(row);
  }
}
