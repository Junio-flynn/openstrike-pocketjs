// Input/focus public API.

export { BTN } from "../spec/spec.ts";
export { touches, type TouchContact } from "./touch.ts";
export {
  cursorX,
  cursorY,
  clearCursorPosition,
  enableCursor,
  focusNode,
  getFocused,
  pushFocusGrid,
  pushFocusScope,
  setCursorPosition,
  type CursorOptions,
  type FocusGridOptions,
  type FocusScopeOptions,
} from "./input.ts";
