import fs from 'node:fs';
import path from 'node:path';
import type { Compiler, Configuration } from 'webpack';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const outDir = path.join(__dirname, '../build/playground');
const template = path.join(__dirname, '../src/playground/index.html');

/**
 * Встраивает собранный скрипт в шаблон и пишет два файла:
 *  - playground.html — фрагмент без <html>/<head> (для публикации как Artifact);
 *  - index.html — полноценный документ, чтобы открыть локально в браузере.
 */
class InlinePlaygroundPlugin {
  apply(compiler: Compiler) {
    // Шаблон читается плагином, а не импортируется, поэтому сообщаем webpack о нём явно:
    // иначе в режиме --watch правки HTML не запускали бы пересборку
    compiler.hooks.afterCompile.tap('InlinePlaygroundPlugin', compilation => {
      compilation.fileDependencies.add(template);
    });

    compiler.hooks.afterEmit.tap('InlinePlaygroundPlugin', () => {
      const script = fs
        .readFileSync(path.join(outDir, 'main.js'), 'utf-8')
        .replace(/<\/script/gi, '<\\/script');
      const page = fs
        .readFileSync(template, 'utf-8')
        .replace('<!-- SCRIPT -->', () => `<script>${script}</script>`);

      fs.writeFileSync(path.join(outDir, 'playground.html'), page);
      fs.writeFileSync(
        path.join(outDir, 'index.html'),
        '<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n' +
          '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' +
          '</head>\n<body>\n' +
          page +
          '\n</body>\n</html>\n',
      );
    });
  }
}

const webpackPlaygroundConfig: Configuration = {
  target: 'web',
  mode: 'production',
  devtool: false,
  entry: {
    main: path.resolve(__dirname, '../src/playground/main.ts'),
  },
  output: {
    path: outDir,
    filename: '[name].js',
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
      },
    ],
  },
  resolve: {
    extensions: ['.ts', '.js', '.json'],
    alias: {
      '~': path.resolve(__dirname, '..', 'src'),
    },
  },
  plugins: [new InlinePlaygroundPlugin()],
};

export default webpackPlaygroundConfig;
