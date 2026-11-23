import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: [
      ".venv/**",
      "node_modules/**",
      "data/**",
      "dist/**",
      "e2e-artifacts/**",
      "eslint.config.mjs",
    ],
  },
  ...tseslint.configs.strictTypeChecked,
  {
    languageOptions: {
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
    },
    rules: {
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-floating-promises": ["error", { ignoreVoid: false }],
      "@typescript-eslint/no-misused-promises": "error",
      "@typescript-eslint/switch-exhaustiveness-check": "error",
    },
  },
);
