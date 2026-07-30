package com.example.trackingapp.file

import android.content.ContentResolver
import android.database.Cursor
import android.net.Uri
import android.provider.OpenableColumns
import com.example.trackingapp.model.ModelInfo
import com.example.trackingapp.model.ModelRepository
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

enum class ModelFileType(
    val suffix: String,
    val component: String,
    val label: String,
) {
    TEMPLATE_PARAM(".ncnn.param", "template", "template param"),
    TEMPLATE_BIN(".ncnn.bin", "template", "template bin"),
    TRACKING_PARAM(".ncnn.param", "tracking", "tracking param"),
    TRACKING_BIN(".ncnn.bin", "tracking", "tracking bin"),
}

data class SelectedModelFile(
    val uri: Uri,
    val displayName: String,
    val declaredSize: Long?,
    val type: ModelFileType,
) {
    val modelStem: String
        get() = displayName.dropLast(type.suffix.length)
}

class ModelImportManager(
    private val contentResolver: ContentResolver,
    private val repository: ModelRepository,
) {
    fun inspect(uri: Uri, expectedType: ModelFileType): SelectedModelFile {
        val cursor = contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE),
            null,
            null,
            null,
        ) ?: error("无法读取所选文件信息")

        cursor.use {
            check(it.moveToFirst()) { "所选文件不存在" }
            val name = it.requiredString(OpenableColumns.DISPLAY_NAME)
            require(name == File(name).name && '/' !in name && '\\' !in name) { "文件名不安全" }
            require(name.endsWith(expectedType.suffix, ignoreCase = true)) {
                "请选择 ${expectedType.label}（${expectedType.suffix}）"
            }
            val stem = name.dropLast(expectedType.suffix.length)
            packageName(stem, expectedType.component)

            val sizeIndex = it.getColumnIndex(OpenableColumns.SIZE)
            val size = if (sizeIndex >= 0 && !it.isNull(sizeIndex)) it.getLong(sizeIndex) else null
            require(size == null || size > 0L) { "所选文件为空" }
            return SelectedModelFile(uri, name, size, expectedType)
        }
    }

    fun importPackage(selections: Map<ModelFileType, SelectedModelFile>): ModelInfo {
        ModelFileType.values().forEach { type ->
            require(selections[type]?.type == type) { "缺少 ${type.label} 文件" }
        }
        val templateParam = selections.getValue(ModelFileType.TEMPLATE_PARAM)
        val templateBin = selections.getValue(ModelFileType.TEMPLATE_BIN)
        val trackingParam = selections.getValue(ModelFileType.TRACKING_PARAM)
        val trackingBin = selections.getValue(ModelFileType.TRACKING_BIN)

        require(templateParam.modelStem.equals(templateBin.modelStem, ignoreCase = true)) {
            "template param 与 bin 的基础文件名必须一致"
        }
        require(trackingParam.modelStem.equals(trackingBin.modelStem, ignoreCase = true)) {
            "tracking param 与 bin 的基础文件名必须一致"
        }
        val templatePackage = packageName(templateParam.modelStem, "template")
        val trackingPackage = packageName(trackingParam.modelStem, "tracking")
        require(templatePackage.equals(trackingPackage, ignoreCase = true)) {
            "template 与 tracking 必须属于同一个模型包"
        }

        val modelId = repository.nextAvailableId(templatePackage)
        val finalDirectory = File(repository.rootDirectory, modelId)
        val stagingDirectory = File(repository.rootDirectory, ".import-${UUID.randomUUID()}")
        check(stagingDirectory.mkdir()) { "无法创建模型导入临时目录" }

        try {
            val templateParamFile = copyAndVerify(templateParam, stagingDirectory)
            val templateBinFile = copyAndVerify(templateBin, stagingDirectory)
            val trackingParamFile = copyAndVerify(trackingParam, stagingDirectory)
            val trackingBinFile = copyAndVerify(trackingBin, stagingDirectory)
            val info = ModelInfo(
                id = modelId,
                name = templatePackage,
                source = ModelInfo.SOURCE_IMPORTED,
                templateParamFileName = templateParamFile.name,
                templateBinFileName = templateBinFile.name,
                trackingParamFileName = trackingParamFile.name,
                trackingBinFileName = trackingBinFile.name,
                templateParamSize = templateParamFile.length(),
                templateBinSize = templateBinFile.length(),
                trackingParamSize = trackingParamFile.length(),
                trackingBinSize = trackingBinFile.length(),
                importedAtMillis = System.currentTimeMillis(),
                directory = finalDirectory,
            )
            File(stagingDirectory, ModelInfo.METADATA_FILE_NAME)
                .writeText(info.toJson().toString(2), Charsets.UTF_8)
            check(stagingDirectory.renameTo(finalDirectory)) { "无法提交已导入模型" }
            return ModelInfo.fromDirectory(finalDirectory)
        } catch (error: Throwable) {
            stagingDirectory.deleteRecursively()
            throw error
        }
    }

    private fun copyAndVerify(selection: SelectedModelFile, directory: File): File {
        val destination = File(directory, selection.displayName)
        val input = contentResolver.openInputStream(selection.uri)
            ?: error("无法打开 ${selection.type.label} 文件")
        input.use { source ->
            FileOutputStream(destination).use { output -> source.copyTo(output) }
        }
        check(destination.length() > 0L) { "${selection.type.label} 文件为空" }
        return destination
    }

    private fun packageName(stem: String, component: String): String {
        val marker = listOf("_$component", "-$component", ".$component")
            .firstOrNull { stem.endsWith(it, ignoreCase = true) }
            ?: throw IllegalArgumentException("文件名必须以 _$component、-$component 或 .$component 结尾")
        val name = stem.dropLast(marker.length)
        require(name.isNotBlank()) { "模型包名称不能为空" }
        return name
    }

    private fun Cursor.requiredString(columnName: String): String {
        val index = getColumnIndex(columnName)
        check(index >= 0 && !isNull(index)) { "文件提供方未返回 $columnName" }
        return getString(index)
    }
}
