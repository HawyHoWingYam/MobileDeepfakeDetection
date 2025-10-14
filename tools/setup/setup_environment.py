#!/usr/bin/env python3
"""
AWARE-NET 環境驗證與自動化設置腳本
Environment Validation and Automated Setup Script

這個腳本用於驗證和自動設置AWARE-NET項目的運行環境
包括依賴檢查、GPU配置、數據集驗證等功能
"""

import os
import sys
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentValidator:
    """AWARE-NET環境驗證器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.system_info = self._get_system_info()
        self.validation_results = {}

    def _get_system_info(self) -> Dict:
        """獲取系統信息"""
        return {
            'platform': platform.platform(),
            'system': platform.system(),
            'machine': platform.machine(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'python_executable': sys.executable
        }

    def validate_python_environment(self) -> Dict:
        """驗證Python環境"""
        logger.info("🐍 驗證Python環境...")

        result = {
            'status': 'pass',
            'version': sys.version_info[:3],
            'executable': sys.executable,
            'issues': []
        }

        # 檢查Python版本
        if sys.version_info < (3, 8):
            result['status'] = 'fail'
            result['issues'].append("Python版本過舊，需要3.8+")
            logger.error(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} (需要 >= 3.8)")
        else:
            logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

        # 檢查pip可用性
        try:
            import pip
            result['pip_available'] = True
            logger.info("✅ pip 可用")
        except ImportError:
            result['pip_available'] = False
            result['issues'].append("pip不可用")
            logger.error("❌ pip 不可用")

        return result

    def validate_pytorch_installation(self) -> Dict:
        """驗證PyTorch安裝"""
        logger.info("🔥 驗證PyTorch安裝...")

        result = {
            'status': 'pass',
            'installed': False,
            'version': None,
            'cuda_available': False,
            'gpu_info': [],
            'issues': []
        }

        try:
            import torch
            result['installed'] = True
            result['version'] = torch.__version__
            result['cuda_available'] = torch.cuda.is_available()

            logger.info(f"✅ PyTorch {torch.__version__}")

            if result['cuda_available']:
                logger.info(f"✅ CUDA {torch.version.cuda}")

                # 獲取GPU信息
                for i in range(torch.cuda.device_count()):
                    gpu_info = {
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory': torch.cuda.get_device_properties(i).total_memory / 1024**3,
                        'capability': torch.cuda.get_device_capability(i)
                    }
                    result['gpu_info'].append(gpu_info)
                    logger.info(f"🎮 GPU {i}: {gpu_info['name']} ({gpu_info['memory']:.1f}GB)")
            else:
                logger.warning("⚠️  CUDA不可用，將使用CPU模式")

            # 測試基本操作
            try:
                x = torch.tensor([1.0, 2.0, 3.0])
                y = x * 2
                assert y.sum().item() == 12.0
                logger.info("✅ PyTorch基本操作正常")
            except Exception as e:
                result['status'] = 'warning'
                result['issues'].append(f"基本操作測試失敗: {e}")
                logger.warning(f"⚠️  基本操作測試失敗: {e}")

        except ImportError:
            result['status'] = 'fail'
            result['installed'] = False
            result['issues'].append("PyTorch未安裝")
            logger.error("❌ PyTorch未安裝")
        except Exception as e:
            result['status'] = 'fail'
            result['issues'].append(f"PyTorch驗證錯誤: {e}")
            logger.error(f"❌ PyTorch驗證錯誤: {e}")

        return result

    def validate_required_packages(self) -> Dict:
        """驗證必需的包"""
        logger.info("📦 驗證必需包...")

        required_packages = {
            'numpy': '數值計算',
            'pandas': '數據處理',
            'sklearn': '機器學習',
            'matplotlib': '繪圖',
            'seaborn': '統計繪圖',
            'cv2': 'OpenCV圖像處理',
            'PIL': 'Pillow圖像處理',
            'tqdm': '進度條',
            'albumentations': '數據增強',
            'timm': '預訓練模型'
        }

        result = {
            'status': 'pass',
            'installed': {},
            'missing': [],
            'issues': []
        }

        for package, description in required_packages.items():
            try:
                if package == 'cv2':
                    import cv2
                    version = cv2.__version__
                elif package == 'sklearn':
                    import sklearn
                    version = sklearn.__version__
                elif package == 'PIL':
                    from PIL import Image
                    version = Image.__version__
                else:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')

                result['installed'][package] = {
                    'version': version,
                    'description': description
                }
                logger.info(f"✅ {package} ({version})")

            except ImportError:
                result['missing'].append(package)
                result['issues'].append(f"缺少{package} ({description})")
                logger.warning(f"⚠️  缺少 {package} - {description}")

        if result['missing']:
            result['status'] = 'warning' if len(result['missing']) <= 2 else 'fail'

        return result

    def validate_project_structure(self) -> Dict:
        """驗證項目結構"""
        logger.info("📁 驗證項目結構...")

        required_dirs = [
            'src',
            'src/stage_00',
            'src/stage_01',
            'src/stage_02',
            'src/utils',
            'configs',
            'scripts'
        ]

        required_files = [
            'environment.yml',
            'setup.py',
            'src/stage_00/baseline_model.py',
            'src/stage_00/train_baseline.py',
            'src/utils/dataset_config.py',
            'src/utils/metrics.py',
            'configs/datasets.json'
        ]

        result = {
            'status': 'pass',
            'directories': {},
            'files': {},
            'missing_dirs': [],
            'missing_files': [],
            'issues': []
        }

        # 檢查目錄
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            exists = full_path.exists() and full_path.is_dir()
            result['directories'][dir_path] = exists

            if exists:
                logger.info(f"✅ 目錄: {dir_path}")
            else:
                result['missing_dirs'].append(dir_path)
                logger.warning(f"⚠️  缺少目錄: {dir_path}")

        # 檢查文件
        for file_path in required_files:
            full_path = self.project_root / file_path
            exists = full_path.exists() and full_path.is_file()
            result['files'][file_path] = exists

            if exists:
                logger.info(f"✅ 文件: {file_path}")
            else:
                result['missing_files'].append(file_path)
                logger.warning(f"⚠️  缺少文件: {file_path}")

        # 評估狀態
        if result['missing_dirs'] or result['missing_files']:
            total_missing = len(result['missing_dirs']) + len(result['missing_files'])
            if total_missing > 3:
                result['status'] = 'fail'
            else:
                result['status'] = 'warning'

            result['issues'].extend([f"缺少目錄: {d}" for d in result['missing_dirs']])
            result['issues'].extend([f"缺少文件: {f}" for f in result['missing_files']])

        return result

    def validate_dataset_configuration(self) -> Dict:
        """驗證數據集配置"""
        logger.info("🗃️ 驗證數據集配置...")

        result = {
            'status': 'pass',
            'config_file_exists': False,
            'paths_valid': {},
            'manifests_exist': False,
            'issues': []
        }

        # 檢查配置文件
        config_path = self.project_root / "configs" / "datasets.json"
        if config_path.exists():
            result['config_file_exists'] = True
            logger.info("✅ 數據集配置文件存在")

            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # 檢查路徑
                if 'paths' in config:
                    for key, path_str in config['paths'].items():
                        path = Path(path_str)
                        exists = path.exists()
                        result['paths_valid'][key] = exists

                        if exists:
                            logger.info(f"✅ 路徑 {key}: {path}")
                        else:
                            logger.warning(f"⚠️  路徑 {key} 不存在: {path}")
                            result['issues'].append(f"數據集路徑不存在: {key}")

            except Exception as e:
                result['issues'].append(f"配置文件讀取錯誤: {e}")
                logger.error(f"❌ 配置文件讀取錯誤: {e}")
        else:
            result['config_file_exists'] = False
            result['issues'].append("數據集配置文件不存在")
            logger.warning("⚠️  數據集配置文件不存在")

        # 檢查manifest文件
        manifests_dir = self.project_root / "manifests"
        if manifests_dir.exists():
            manifest_files = list(manifests_dir.glob("*.csv"))
            result['manifests_exist'] = len(manifest_files) > 0

            if result['manifests_exist']:
                logger.info(f"✅ 找到 {len(manifest_files)} 個manifest文件")
            else:
                logger.warning("⚠️  manifests目錄存在但無CSV文件")
        else:
            logger.warning("⚠️  manifests目錄不存在")

        # 評估狀態
        if result['issues']:
            result['status'] = 'warning' if len(result['issues']) <= 2 else 'fail'

        return result

    def run_full_validation(self) -> Dict:
        """運行完整驗證"""
        logger.info("🔍 開始AWARE-NET環境完整驗證...")
        logger.info("=" * 60)

        # 運行各項驗證
        validations = {
            'python': self.validate_python_environment(),
            'pytorch': self.validate_pytorch_installation(),
            'packages': self.validate_required_packages(),
            'project': self.validate_project_structure(),
            'dataset': self.validate_dataset_configuration()
        }

        # 計算整體狀態
        overall_status = self._calculate_overall_status(validations)

        # 生成報告
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': self.system_info,
            'validations': validations,
            'overall_status': overall_status,
            'recommendations': self._generate_recommendations(validations)
        }

        self.validation_results = report
        return report

    def _calculate_overall_status(self, validations: Dict) -> Dict:
        """計算整體狀態"""
        statuses = [v['status'] for v in validations.values()]

        if all(s == 'pass' for s in statuses):
            status = 'excellent'
            message = "🎉 環境配置完美！"
        elif 'fail' not in statuses:
            status = 'good'
            message = "✅ 環境配置良好，有少量警告"
        elif statuses.count('fail') == 1:
            status = 'fair'
            message = "⚠️ 環境配置尚可，需要解決關鍵問題"
        else:
            status = 'poor'
            message = "❌ 環境配置需要大量改進"

        return {
            'level': status,
            'message': message,
            'pass_count': statuses.count('pass'),
            'warning_count': statuses.count('warning'),
            'fail_count': statuses.count('fail')
        }

    def _generate_recommendations(self, validations: Dict) -> List[str]:
        """生成建議"""
        recommendations = []

        # Python環境建議
        if validations['python']['status'] == 'fail':
            recommendations.append("升級Python到3.8或更高版本")

        # PyTorch建議
        if validations['pytorch']['status'] == 'fail':
            recommendations.append("運行 python setup.py 安裝PyTorch")
        elif not validations['pytorch']['cuda_available']:
            recommendations.append("考慮安裝CUDA版本的PyTorch以獲得更好性能")

        # 包依賴建議
        if validations['packages']['missing']:
            missing = validations['packages']['missing']
            if len(missing) <= 3:
                recommendations.append(f"安裝缺少的包: pip install {' '.join(missing)}")
            else:
                recommendations.append("運行 python setup.py 安裝所有依賴")

        # 項目結構建議
        if validations['project']['missing_files']:
            recommendations.append("檢查項目完整性，可能需要重新克隆代碼庫")

        # 數據集建議
        if not validations['dataset']['config_file_exists']:
            recommendations.append("配置數據集路徑: 編輯 configs/datasets.json")
        elif not validations['dataset']['manifests_exist']:
            recommendations.append("運行 python setup.py 準備數據集")

        return recommendations

    def print_validation_report(self):
        """打印驗證報告"""
        if not self.validation_results:
            logger.error("尚未運行驗證，請先調用 run_full_validation()")
            return

        report = self.validation_results

        print("\n" + "=" * 60)
        print("📊 AWARE-NET 環境驗證報告")
        print("=" * 60)

        # 基本信息
        print(f"驗證時間: {report['timestamp']}")
        print(f"系統: {report['system_info']['platform']}")
        print(f"Python: {report['system_info']['python_version']}")

        # 整體狀態
        overall = report['overall_status']
        print(f"\n{overall['message']}")
        print(f"通過: {overall['pass_count']}, 警告: {overall['warning_count']}, 失敗: {overall['fail_count']}")

        # 詳細結果
        print("\n📋 詳細結果:")
        for category, result in report['validations'].items():
            status_icons = {'pass': '✅', 'warning': '⚠️', 'fail': '❌'}
            icon = status_icons.get(result['status'], '❓')
            print(f"{icon} {category.title()}: {result['status']}")

            if result.get('issues'):
                for issue in result['issues'][:3]:  # 最多顯示3個問題
                    print(f"    - {issue}")

        # 建議
        if report['recommendations']:
            print("\n💡 建議:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")

        print("\n" + "=" * 60)

    def save_report(self, filepath: Optional[str] = None):
        """保存報告到文件"""
        if not self.validation_results:
            logger.error("尚未運行驗證")
            return

        if filepath is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filepath = f"environment_validation_{timestamp}.json"

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
            logger.info(f"📄 驗證報告已保存: {filepath}")
        except Exception as e:
            logger.error(f"保存報告失敗: {e}")

def main():
    """主函數"""
    print("🔍 AWARE-NET 環境驗證工具")
    print("=" * 40)

    validator = EnvironmentValidator()

    # 運行完整驗證
    try:
        report = validator.run_full_validation()
        validator.print_validation_report()

        # 詢問是否保存報告
        save_report = input("\n是否保存驗證報告? (y/N): ").strip().lower()
        if save_report == 'y':
            validator.save_report()

        # 根據結果給出下一步建議
        overall_status = report['overall_status']['level']
        print(f"\n🎯 下一步:")

        if overall_status == 'excellent':
            print("✅ 環境準備完成！可以開始訓練:")
            print("   python train.py")
        elif overall_status == 'good':
            print("✅ 環境基本準備完成，建議解決警告後開始訓練")
            print("   python train.py")
        else:
            print("❌ 請根據上述建議修復環境問題")
            print("   python setup.py  # 運行主設置腳本")

    except KeyboardInterrupt:
        print("\n⏹️ 驗證被用戶中斷")
    except Exception as e:
        logger.error(f"驗證過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
